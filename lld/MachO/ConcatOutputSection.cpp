//===- ConcatOutputSection.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConcatOutputSection.h"
#include "Config.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/CommonLinkerContext.h"
#include "llvm/BinaryFormat/MachO.h"
#include <deque>

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

MapVector<NamePair, ConcatOutputSection *> macho::concatOutputSections;

void ConcatOutputSection::addInput(ConcatInputSection *input) {
  assert(input->parent == this);
  if (inputs.empty()) {
    align = input->align;
    flags = input->getFlags();
  } else {
    align = std::max(align, input->align);
    finalizeFlags(input);
  }
  inputs.push_back(input);
}

// Branch-range extension can be implemented in two ways, either through ...
//
// (1) Branch islands: Single branch instructions (also of limited range),
//     that might be chained in multiple hops to reach the desired
//     destination. On ARM64, as 16 branch islands are needed to hop between
//     opposite ends of a 2 GiB program. LD64 uses branch islands exclusively,
//     even when it needs excessive hops.
//
// (2) Thunks: Instruction(s) to load the destination address into a scratch
//     register, followed by a register-indirect branch. Thunks are
//     constructed to reach any arbitrary address, so need not be
//     chained. Although thunks need not be chained, a program might need
//     multiple thunks to the same destination distributed throughout a large
//     program so that all call sites can have one within range.
//
// The optimal approach is to mix islands for destinations within two hops,
// and use thunks for destinations at greater distance. For now, we only
// implement thunks. TODO: Adding support for branch islands!

DenseMap<ThunkKey, ThunkInfo, ThunkMapKeyInfo> lld::macho::thunkMap;

// Determine whether we need thunks, which depends on the target arch -- RISC
// (i.e., ARM) generally does because it has limited-range branch/call
// instructions, whereas CISC (i.e., x86) generally doesn't. RISC only needs
// thunks for programs so large that branch source & destination addresses
// might differ more than the range of branch instruction(s).
bool TextOutputSection::needsThunks() const {
  if (!target->usesThunks())
    return false;
  uint64_t isecAddr = addr;
  for (ConcatInputSection *isec : inputs)
    isecAddr = alignToPowerOf2(isecAddr, isec->align) + isec->getSize();
  // Other sections besides __text might be small enough to pass this
  // test but nevertheless need thunks for calling into other sections.
  // An imperfect heuristic to use in this case is that if a section
  // we've already processed in this segment needs thunks, so do the
  // rest.
  bool needsThunks = parent && parent->needsThunks;

  // Calculate the total size of all branch target sections
  uint64_t branchTargetsSize = in.stubs->getSize();

  // Add the size of __objc_stubs section if it exists
  if (in.objcStubs && in.objcStubs->isNeeded())
    branchTargetsSize += in.objcStubs->getSize();

  if (!needsThunks &&
      isecAddr - addr + branchTargetsSize <=
          std::min(target->backwardBranchRange, target->forwardBranchRange))
    return false;
  // Yes, this program is large enough to need thunks.
  if (parent)
    parent->needsThunks = true;
  return true;
}

void ConcatOutputSection::finalizeOne(ConcatInputSection *isec) {
  size = alignToPowerOf2(size, isec->align);
  fileSize = alignToPowerOf2(fileSize, isec->align);
  isec->outSecOff = size;
  isec->isFinal = true;
  size += isec->getSize();
  fileSize += isec->getFileSize();
}

void ConcatOutputSection::finalizeContents() {
  for (ConcatInputSection *isec : inputs)
    finalizeOne(isec);
}

bool TextOutputSection::isTargetKnownInRange(const ConcatInputSection &isec,
                                             const Relocation &r) const {
  uint64_t callVA = isec.getVA() + r.offset;
  uint64_t lowVA = target->backwardBranchRange < callVA
                       ? callVA - target->backwardBranchRange
                       : 0;
  uint64_t highVA = callVA + target->forwardBranchRange;
  auto *funcSym = cast<Symbol *>(r.referent);
  uint64_t funcVA = resolveSymbolOffsetVA(funcSym, r.type, r.addend);
  // Check if the referent is reachable with a simple call instruction.
  return lowVA <= funcVA && funcVA <= highVA;
}

Defined *TextOutputSection::getThunkInRange(const ConcatInputSection &isec,
                                            const Relocation &r,
                                            const ThunkInfo &thunkInfo) const {
  assert(!isTargetKnownInRange(isec, r));
  if (!thunkInfo.sym)
    return nullptr;
  uint64_t callVA = isec.getVA() + r.offset;
  uint64_t lowVA = target->backwardBranchRange < callVA
                       ? callVA - target->backwardBranchRange
                       : 0;
  uint64_t highVA = callVA + target->forwardBranchRange;
  uint64_t thunkVA = thunkInfo.isec->getVA();
  if (lowVA <= thunkVA && thunkVA <= highVA)
    return thunkInfo.sym;
  return nullptr;
}

void TextOutputSection::updateBranchTargetToThunk(Relocation &r,
                                                  Defined *thunk) {
  r.referent = thunk;
  // The thunk itself bakes in the addend, so the call-site reloc must
  // branch to the thunk start with no extra offset.
  r.addend = 0;
  ++thunkCallCount;
}

void TextOutputSection::createThunk(const ConcatInputSection &isec,
                                    Relocation &r, ThunkInfo &thunkInfo) {
  assert(getThunkInRange(isec, r, thunkInfo) == nullptr);
  assert(isec.isFinal);
  uint64_t highVA = isec.getVA() + r.offset + target->forwardBranchRange;
  if (addr + size > highVA) {
    // There were too many consecutive branch instructions for `slop`
    // below. If you hit this: For the current algorithm, just bumping up
    // slop below and trying again is probably simplest. (See also PR51578
    // comment 5).
    fatal(Twine(__FUNCTION__) +
          ": FIXME: thunk range overrun. Consider increasing the "
          "slop-scale with `--slop-scale=<unsigned_int>`.");
  }
  thunkInfo.isec = makeSyntheticInputSection(isec.getSegName(), isec.getName());
  thunkInfo.isec->parent = this;
  assert(thunkInfo.isec->live);

  std::string addendSuffix;
  if (r.addend != 0)
    addendSuffix = "+" + std::to_string(r.addend);
  size_t thunkSize = target->thunkSize;
  auto *funcSym = cast<Symbol *>(r.referent);
  StringRef thunkName =
      saver().save(funcSym->getName() + addendSuffix + ".thunk." +
                   std::to_string(thunkInfo.sequence++));
  if (!isa<Defined>(funcSym) || cast<Defined>(funcSym)->isExternal()) {
    thunkInfo.sym = symtab->addDefined(
        thunkName, /*file=*/nullptr, thunkInfo.isec, /*value=*/0, thunkSize,
        /*isWeakDef=*/false, /*isPrivateExtern=*/true,
        /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
        /*isWeakDefCanBeHidden=*/false);
  } else {
    thunkInfo.sym = make<Defined>(
        thunkName, /*file=*/nullptr, thunkInfo.isec, /*value=*/0, thunkSize,
        /*isWeakDef=*/false, /*isExternal=*/false, /*isPrivateExtern=*/true,
        /*includeInSymtab=*/true, /*isReferencedDynamically=*/false,
        /*noDeadStrip=*/false, /*isWeakDefCanBeHidden=*/false);
  }
  thunkInfo.sym->used = true;
  target->populateThunk(thunkInfo.isec, funcSym, r.addend);
  updateBranchTargetToThunk(r, thunkInfo.sym);
  finalizeOne(thunkInfo.isec);
  thunks.push_back(thunkInfo.isec);
}

bool TextOutputSection::isTargetStubsAndInRange(
    const ConcatInputSection &isec, const Relocation &r,
    uint64_t estimatedStubsEnd) const {
  auto *funcSym = cast<Symbol *>(r.referent);
  if (!funcSym->isInStubs() && !(in.objcStubs && in.objcStubs->isNeeded() &&
                                 ObjCStubsSection::isObjCStubSymbol(funcSym)))
    return false;
  if (r.addend)
    return false;
  uint64_t highVA = isec.getVA() + r.offset + target->forwardBranchRange;
  return estimatedStubsEnd <= highVA;
}

void TextOutputSection::finalize() {
  if (!needsThunks()) {
    for (ConcatInputSection *isec : inputs)
      finalizeOne(isec);
    return;
  }

  // Branches whose target sections are out of range or have not yet been
  // finalized. We may need to emit thunks for them.
  std::deque<std::pair<ConcatInputSection *, Relocation *>> branchesToProcess;
  // Branches whose targets have not yet be finalized, but a thunk for that
  // target exists. We defer processing these branches because it's possible we
  // can still direct call to their targets after they have all been finalized.
  SmallVector<std::tuple<ConcatInputSection *, Relocation *, Defined *>>
      deferredBranchRedirects;

  const uint64_t slop = config->slopScale * target->thunkSize;
  for (auto *isec : inputs) {
    while (!branchesToProcess.empty()) {
      auto [callerIsec, r] = branchesToProcess.front();
      assert(callerIsec->isFinal);
      auto &thunkInfo = thunkMap[*r];
      if (isTargetKnownInRange(*callerIsec, *r)) {
        branchesToProcess.pop_front();
        continue;
      }
      if (auto *thunk = getThunkInRange(*callerIsec, *r, thunkInfo)) {
        deferredBranchRedirects.emplace_back(callerIsec, r, thunk);
        branchesToProcess.pop_front();
        continue;
      }
      uint64_t highVA =
          callerIsec->getVA() + r->offset + target->forwardBranchRange;
      uint64_t nextEnd =
          alignToPowerOf2(addr + size, isec->align) + isec->getSize();
      // If we were to emit this section, would we have enough space for more
      // thunks? If we do, then we can delay processing this thunk so we may
      // finalize more potencial target sections. Otherwise we must emit thunks
      // until we have enough space.
      if (nextEnd + slop <= highVA)
        break;

      createThunk(*callerIsec, *r, thunkInfo);
      branchesToProcess.pop_front();
    }
    finalizeOne(isec);

    // TODO: Remove this check and the assert below. In fact, I don't believe
    // the relocation iteration order matters for correctness.
    bool hasCallsite = llvm::any_of(isec->relocs, [](Relocation &r) {
      return target->hasAttr(r.type, RelocAttrBits::BRANCH);
    });
    if (!hasCallsite)
      continue;

    // Process relocs by ascending address, i.e., ascending offset within isec
    // FIXME: This property does not hold for object files produced by ld64's
    // `-r` mode.
    assert(is_sorted(isec->relocs, [](Relocation &a, Relocation &b) {
      return a.offset > b.offset;
    }));
    for (Relocation &r : reverse(isec->relocs)) {
      if (!target->hasAttr(r.type, RelocAttrBits::BRANCH))
        continue;
      if (isTargetKnownInRange(*isec, r))
        continue;
      auto &thunkInfo = thunkMap[r];
      if (auto *thunk = getThunkInRange(*isec, r, thunkInfo)) {
        deferredBranchRedirects.emplace_back(isec, &r, thunk);
        continue;
      }
      branchesToProcess.emplace_back(isec, &r);
    }
  }

  llvm::erase_if(branchesToProcess, [&](auto &pair) {
    auto [callerIsec, r] = pair;
    return isTargetKnownInRange(*callerIsec, *r);
  });
  // Count distinct unresolved branch targets that still lack an in-range thunk.
  // We use this as an upper bound on the number of thunks we may still create
  // when estimating where __stubs / __objc_stubs could end up.
  DenseSet<ThunkKey, ThunkMapKeyInfo> branchTargets;
  for (auto [callerIsec, r] : branchesToProcess) {
    ThunkKey thunkKey(*r);
    auto &thunkInfo = thunkMap[thunkKey];
    if (!getThunkInRange(*callerIsec, *r, thunkInfo))
      branchTargets.insert(thunkKey);
  }

  uint64_t estimatedTextEnd =
      addr + size + branchTargets.size() * target->thunkSize;
  uint64_t estimatedStubsEnd =
      alignToPowerOf2(estimatedTextEnd, in.stubs->align) + in.stubs->getSize();
  if (in.objcStubs && in.objcStubs->isNeeded())
    estimatedStubsEnd =
        alignToPowerOf2(estimatedStubsEnd, in.objcStubs->align) +
        in.objcStubs->getSize();

  for (auto [isec, r, thunk] : deferredBranchRedirects) {
    if (isTargetKnownInRange(*isec, *r))
      continue;
    if (isTargetStubsAndInRange(*isec, *r, estimatedStubsEnd))
      continue;
    updateBranchTargetToThunk(*r, thunk);
  }

  for (auto [isec, r] : branchesToProcess) {
    if (isTargetStubsAndInRange(*isec, *r, estimatedStubsEnd))
      continue;
    auto &thunkInfo = thunkMap[*r];
    if (auto *thunk = getThunkInRange(*isec, *r, thunkInfo)) {
      updateBranchTargetToThunk(*r, thunk);
      continue;
    }
    createThunk(*isec, *r, thunkInfo);
  }

  if (!thunks.empty())
    log(name + ": Created " + Twine(thunks.size()) + " (" +
        Twine(thunks.size() * target->thunkSize / 1024) +
        " KB) thunks and updated " + Twine(thunkCallCount) + " branch targets");
}

void ConcatOutputSection::writeTo(uint8_t *buf) const {
  for (ConcatInputSection *isec : inputs)
    isec->writeTo(buf + isec->outSecOff);
}

void TextOutputSection::writeTo(uint8_t *buf) const {
  // Merge input sections from thunk & ordinary vectors
  size_t i = 0, ie = inputs.size();
  size_t t = 0, te = thunks.size();
  while (i < ie || t < te) {
    while (i < ie && (t == te || inputs[i]->empty() ||
                      inputs[i]->outSecOff < thunks[t]->outSecOff)) {
      inputs[i]->writeTo(buf + inputs[i]->outSecOff);
      ++i;
    }
    while (t < te && (i == ie || thunks[t]->outSecOff < inputs[i]->outSecOff)) {
      thunks[t]->writeTo(buf + thunks[t]->outSecOff);
      ++t;
    }
  }
}

void ConcatOutputSection::finalizeFlags(InputSection *input) {
  switch (sectionType(input->getFlags())) {
  default /*type-unspec'ed*/:
    // FIXME: Add additional logic here when supporting emitting obj files.
    break;
  case S_4BYTE_LITERALS:
  case S_8BYTE_LITERALS:
  case S_16BYTE_LITERALS:
  case S_CSTRING_LITERALS:
  case S_ZEROFILL:
  case S_LAZY_SYMBOL_POINTERS:
  case S_MOD_TERM_FUNC_POINTERS:
  case S_THREAD_LOCAL_REGULAR:
  case S_THREAD_LOCAL_ZEROFILL:
  case S_THREAD_LOCAL_VARIABLES:
  case S_THREAD_LOCAL_INIT_FUNCTION_POINTERS:
  case S_THREAD_LOCAL_VARIABLE_POINTERS:
  case S_NON_LAZY_SYMBOL_POINTERS:
  case S_SYMBOL_STUBS:
    flags |= input->getFlags();
    break;
  }
}

ConcatOutputSection *
ConcatOutputSection::getOrCreateForInput(const InputSection *isec) {
  NamePair names = maybeRenameSection({isec->getSegName(), isec->getName()});
  ConcatOutputSection *&osec = concatOutputSections[names];
  if (!osec) {
    if (isec->getSegName() == segment_names::text &&
        isec->getName() != section_names::gccExceptTab &&
        isec->getName() != section_names::ehFrame)
      osec = make<TextOutputSection>(names.second);
    else
      osec = make<ConcatOutputSection>(names.second);
  }
  return osec;
}

NamePair macho::maybeRenameSection(NamePair key) {
  auto newNames = config->sectionRenameMap.find(key);
  if (newNames != config->sectionRenameMap.end())
    return newNames->second;
  return key;
}
