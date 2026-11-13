//===- ObjCMethodLists.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Find relative method lists through ObjC metadata, not symbol spellings.
// Only whole eligible input sections are converted; embedded records and
// exposed interior addresses stay absolute. No aggregate splitting is needed.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "ObjC.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <array>

using namespace llvm;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::macho;

namespace {
using Ref = std::pair<ConcatInputSection *, uint64_t>;

enum class MetadataKind {
  Class,
  ClassRO,
  Category,
  StubCategory,
  Protocol,
  ProtocolList
};

class MethodListCollector {
public:
  void run();

private:
  std::pair<InputSection *, uint64_t> targetOf(const Relocation &r) const;
  bool isLocalPointer(const Relocation *r) const;
  const Relocation *relocAt(const ConcatInputSection *isec,
                            uint64_t offset) const;
  Ref follow(Ref from, uint64_t field) const;
  void enqueue(MetadataKind kind, Ref ref);
  void visitClass(Ref cls);
  void visitClassRO(Ref ro);
  void visitCategory(Ref cat, bool followClass);
  void visitProtocol(Ref proto);
  void visitProtocolList(Ref list);
  void addCandidate(Ref list, bool isProtocol = false);
  bool isEligible(const ConcatInputSection *isec, bool isProtocol) const;
  void dropInteriorReferences();
  void moveToMethListSection();

  const uint64_t wordSize = target->wordSize;
  std::array<DenseSet<Ref>, 6> visited;
  SmallVector<std::pair<MetadataKind, Ref>, 32> worklist;
  SetVector<ConcatInputSection *> lists;
  mutable DenseMap<const ConcatInputSection *,
                   DenseMap<uint32_t, const Relocation *>>
      relocations;
};
} // namespace

std::pair<InputSection *, uint64_t>
MethodListCollector::targetOf(const Relocation &r) const {
  InputSection *isec = nullptr;
  uint64_t value = 0;
  if (auto *sym = dyn_cast<Symbol *>(r.referent)) {
    auto *def = dyn_cast<Defined>(sym);
    if (!def)
      return {};
    isec = def->isec();
    value = def->value;
  } else {
    isec = dyn_cast<InputSection *>(r.referent);
  }
  if (!isec)
    return {};
  int64_t offset;
  if (value > INT64_MAX ||
      llvm::AddOverflow(static_cast<int64_t>(value), r.getAddend(), offset) ||
      offset < 0 || static_cast<uint64_t>(offset) > isec->data.size())
    return {};
  return {isec, static_cast<uint64_t>(offset)};
}

// An absolute, pointer-sized reference to local data: a defined symbol that
// needs no binding, or a section. Signed (arm64e) pointers are accepted here;
// isConvertibleAuth() decides which of them a converted list may contain.
bool MethodListCollector::isLocalPointer(const Relocation *r) const {
  if (!r || r->pcrel || r->length != target->p2WordSize ||
      !target->hasAttr(r->type, RelocAttrBits::UNSIGNED))
    return false;
  if (auto *sym = dyn_cast<Symbol *>(r->referent))
    return isa<Defined>(sym) && !needsBinding(sym);
  return isa<InputSection *>(r->referent);
}

const Relocation *MethodListCollector::relocAt(const ConcatInputSection *isec,
                                               uint64_t offset) const {
  if (offset > UINT32_MAX)
    return nullptr;
  auto [it, inserted] = relocations.try_emplace(isec);
  if (inserted) {
    it->second.reserve(isec->relocs.size());
    for (const Relocation &r : isec->relocs) {
      auto [entry, unique] = it->second.try_emplace(r.offset, &r);
      if (!unique)
        entry->second = nullptr; // Ambiguous, not a plain pointer.
    }
  }
  return it->second.lookup(offset);
}

Ref MethodListCollector::follow(Ref from, uint64_t field) const {
  auto [isec, offset] = from;
  if (!isec || isec->shouldOmitFromOutput() || isZeroFill(isec->getFlags()) ||
      !isec->data.data() || offset > isec->data.size() ||
      field > isec->data.size() - offset ||
      wordSize > isec->data.size() - offset - field)
    return {};
  const Relocation *r = relocAt(isec, offset + field);
  if (!r || r->pcrel || r->length != target->p2WordSize ||
      !target->hasAttr(r->type, RelocAttrBits::UNSIGNED))
    return {};
  // Roots may name locally defined weak metadata even if dyld binds the root.
  auto [dest, destOffset] = targetOf(*r);
  auto *concat = dyn_cast_or_null<ConcatInputSection>(dest);
  if (!concat || concat->shouldOmitFromOutput())
    return {};
  return {concat, destOffset};
}

void MethodListCollector::enqueue(MetadataKind kind, Ref ref) {
  if (!ref.first ||
      ((kind == MetadataKind::Protocol || kind == MetadataKind::ProtocolList) &&
       !config->enableObjCProtocolRelativeMethodLists))
    return;
  if (visited[static_cast<size_t>(kind)].insert(ref).second)
    worklist.emplace_back(kind, ref);
}

// class_t: isa, superclass, cache, vtable, data. The low bits of data are
// flags; the rest points to the class_ro_t.
void MethodListCollector::visitClass(Ref cls) {
  Ref ro = follow(cls, 4 * wordSize);
  if (ro.first) {
    ro.second &= ~(wordSize - 1);
    enqueue(MetadataKind::ClassRO, ro);
  }
  enqueue(MetadataKind::Class, follow(cls, 0));
}

// class_ro_t: flags, instanceStart, instanceSize, reserved (LP64 only), then
// ivarLayout, name, baseMethods, baseProtocols, ...
void MethodListCollector::visitClassRO(Ref ro) {
  uint64_t header = wordSize == 8 ? 16 : 12;
  addCandidate(follow(ro, header + 2 * wordSize));
  enqueue(MetadataKind::ProtocolList, follow(ro, header + 3 * wordSize));
}

// category_t: name, cls, instanceMethods, classMethods, protocols, ...
void MethodListCollector::visitCategory(Ref cat, bool followClass) {
  addCandidate(follow(cat, 2 * wordSize));
  addCandidate(follow(cat, 3 * wordSize));
  enqueue(MetadataKind::ProtocolList, follow(cat, 4 * wordSize));
  // __objc_catlist2 names a Swift class stub, not a class_t.
  if (followClass)
    enqueue(MetadataKind::Class, follow(cat, wordSize));
}

// protocol_t: isa, mangledName, protocols, instanceMethods, classMethods,
// optionalInstanceMethods, optionalClassMethods, ...
void MethodListCollector::visitProtocol(Ref proto) {
  enqueue(MetadataKind::ProtocolList, follow(proto, 2 * wordSize));
  for (uint64_t field = 3; field != 7; ++field)
    addCandidate(follow(proto, field * wordSize), /*isProtocol=*/true);
}

// protocol_list_t: a pointer-sized count, then that many protocol_t pointers.
void MethodListCollector::visitProtocolList(Ref list) {
  auto [isec, offset] = list;
  if (isZeroFill(isec->getFlags()) || !isec->data.data() ||
      offset > isec->data.size() || isec->data.size() - offset < wordSize)
    return;
  uint64_t count = wordSize == 8 ? read64le(isec->data.data() + offset)
                                 : read32le(isec->data.data() + offset);
  if (count > (isec->data.size() - offset - wordSize) / wordSize)
    return;
  for (uint64_t i = 1; i <= count; ++i)
    enqueue(MetadataKind::Protocol, follow(list, i * wordSize));
}

void MethodListCollector::addCandidate(Ref list, bool isProtocol) {
  if (list.first && list.second == 0 && isEligible(list.first, isProtocol))
    lists.insert(list.first);
}

// arm64e signs method-list IMPs with the ObjC method-list schema (IA key,
// address diversity, zero discriminator). A relative IMP is unsigned and the
// runtime signs it on use, so only that schema can be converted. Any other
// signed field keeps the list absolute.
static bool isConvertibleAuth(const Relocation &r, bool isImp) {
  const AuthInfo *auth = r.getAuthInfo();
  if (!auth)
    return !target->hasAttr(r.type, RelocAttrBits::AUTH);
  return isImp && auth->key == PtrAuthKeyIA && auth->addrDiv &&
         auth->diversity == 0;
}

bool MethodListCollector::isEligible(const ConcatInputSection *isec,
                                     bool isProtocol) const {
  if (lists.contains(const_cast<ConcatInputSection *>(isec)))
    return false;
  StringRef name = isec->getName();
  if ((name != section_names::objcConst && name != section_names::objcData) ||
      sectionType(isec->getFlags()) != MachO::S_REGULAR ||
      isCodeSection(isec) || isec->shouldOmitFromOutput() ||
      !isec->data.data() || isec->data.size() < 8)
    return false;
  // method_list_t: entsizeAndFlags, count, then {name, types, imp} entries.
  // The high 16 bits and low 2 bits of entsizeAndFlags are flags (such as the
  // relative-list bit); an input list to convert has none of them set.
  uint32_t flags = read32le(isec->data.data());
  uint32_t count = read32le(isec->data.data() + 4);
  uint64_t entrySize = 3 * wordSize;
  if ((flags & 0xffff0003) != 0 || (flags & 0xfffc) != entrySize ||
      isec->data.size() != 8 + uint64_t(count) * entrySize)
    return false;
  for (const Defined *sym : isec->symbols)
    if (sym->value != 0 || (sym->size && sym->size != isec->data.size()))
      return false;
  DenseSet<uint32_t> offsets;
  for (const Relocation &r : isec->relocs)
    if (r.offset < 8 || (r.offset - 8) % wordSize != 0 ||
        uint64_t(r.offset) + wordSize > isec->data.size() ||
        !isLocalPointer(&r) || !offsets.insert(r.offset).second ||
        !isConvertibleAuth(r, (r.offset - 8) % entrySize == 2 * wordSize))
      return false;
  for (uint64_t entry = 8; entry < isec->data.size(); entry += entrySize) {
    const Relocation *sel = relocAt(isec, entry);
    if (!isLocalPointer(sel))
      return false;
    auto [selIsec, selOff] = targetOf(*sel);
    auto *strings = dyn_cast_or_null<CStringInputSection>(selIsec);
    if (!strings || strings->getName() != section_names::objcMethname ||
        selOff >= strings->data.size() ||
        strings->getStringPiece(selOff).inSecOff != selOff)
      return false;
    if (auto *sym = dyn_cast<Symbol *>(sel->referent)) {
      uint64_t value = cast<Defined>(sym)->value;
      if (value >= strings->data.size() ||
          strings->getStringPiece(value).inSecOff != selOff)
        return false; // Do not reinterpret a cross-piece symbol addend.
    }
    const Relocation *types = relocAt(isec, entry + wordSize);
    if (!isLocalPointer(types))
      return false;
    auto [typesIsec, typesOff] = targetOf(*types);
    if (!isa_and_nonnull<CStringInputSection>(typesIsec) ||
        typesOff >= typesIsec->data.size())
      return false;
    const Relocation *imp = relocAt(isec, entry + 2 * wordSize);
    if (isProtocol && !imp &&
        llvm::all_of(isec->data.slice(entry + 2 * wordSize, wordSize),
                     [](uint8_t b) { return b == 0; }))
      continue;
    if (!isLocalPointer(imp))
      return false;
  }
  return true;
}

void MethodListCollector::dropInteriorReferences() {
  if (lists.empty())
    return;
  DenseSet<ConcatInputSection *> drop;
  // A symbol at the original section end can be split into a zero-length
  // atom. A live reference to that atom exposes the preceding list's boundary
  // even though its resolved relocation offset is zero.
  DenseMap<InputSection *, ConcatInputSection *> endAtoms;
  for (InputFile *file : inputFiles)
    for (Section *section : file->sections)
      for (size_t i = 1; i < section->subsections.size(); ++i) {
        const Subsection &previous = section->subsections[i - 1];
        const Subsection &next = section->subsections[i];
        auto *list = dyn_cast<ConcatInputSection>(previous.isec);
        if (list && !list->data.empty() && next.isec->data.empty() &&
            previous.offset + list->data.size() == next.offset) {
          endAtoms.try_emplace(next.isec, list);
          // Exported/dynamically observable endpoints need no internal
          // relocation to expose this boundary (for example, dlsym users).
          if (!shouldOmitFromOutput(next.isec) &&
              llvm::any_of(next.isec->symbols, [](const Defined *sym) {
                return sym->isLive() &&
                       ((sym->isExternal() && !sym->privateExtern) ||
                        sym->referencedDynamically);
              }))
            drop.insert(list);
        }
      }
  for (const ConcatInputSection *isec : inputSections) {
    if (isec->shouldOmitFromOutput())
      continue;
    for (const Relocation &r : isec->relocs) {
      auto [dest, offset] = targetOf(r);
      auto *concat = dyn_cast_or_null<ConcatInputSection>(dest);
      if (concat && offset != 0 && lists.contains(concat))
        drop.insert(concat);
      if (ConcatInputSection *list = endAtoms.lookup(dest))
        if (lists.contains(list))
          drop.insert(list);
    }
  }
  lists.remove_if([&](ConcatInputSection *isec) { return drop.count(isec); });
}

void MethodListCollector::moveToMethListSection() {
  llvm::erase_if(inputSections, [&](ConcatInputSection *isec) {
    if (!lists.contains(isec))
      return false;
    isec->parent = in.objcMethList;
    in.objcMethList->addInput(isec);
    in.objcMethList->inputOrder =
        std::min<int>(in.objcMethList->inputOrder, isec->outSecOff);
    return true;
  });
}

void MethodListCollector::run() {
  for (ConcatInputSection *isec : inputSections) {
    if (isec->shouldOmitFromOutput())
      continue;
    StringRef name = isec->getName();
    MetadataKind kind;
    if (name == section_names::objcClassList ||
        name == section_names::objcNonLazyClassList)
      kind = MetadataKind::Class;
    else if (name == section_names::objcClassROList)
      kind = MetadataKind::ClassRO;
    else if (name == section_names::objcCatList ||
             name == section_names::objcNonLazyCatList)
      kind = MetadataKind::Category;
    else if (name == section_names::objcCatList2)
      kind = MetadataKind::StubCategory;
    else if (name == section_names::objcProtoList ||
             name == section_names::objcProtoRefs)
      kind = MetadataKind::Protocol;
    else
      continue;
    for (uint64_t off = 0; off + wordSize <= isec->data.size(); off += wordSize)
      enqueue(kind, follow({isec, 0}, off));
  }
  // An explicit worklist avoids stack growth on deeply inherited protocols.
  while (!worklist.empty()) {
    auto [kind, ref] = worklist.pop_back_val();
    switch (kind) {
    case MetadataKind::Class:
      visitClass(ref);
      break;
    case MetadataKind::ClassRO:
      visitClassRO(ref);
      break;
    case MetadataKind::Category:
      visitCategory(ref, true);
      break;
    case MetadataKind::StubCategory:
      visitCategory(ref, false);
      break;
    case MetadataKind::Protocol:
      visitProtocol(ref);
      break;
    case MetadataKind::ProtocolList:
      visitProtocolList(ref);
      break;
    }
  }
  dropInteriorReferences();
  moveToMethListSection();
}

void objc::collectMethodLists() {
  if (config->emitRelativeMethodLists &&
      config->objcMethodListMetadataDiscovery)
    MethodListCollector().run();
}
