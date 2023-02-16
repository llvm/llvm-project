//===- ObjC.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjC.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "Layout.h"
#include "OutputSegment.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Bitcode/BitcodeReader.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

template <class LP> static bool objectHasObjCSection(MemoryBufferRef mb) {
  using SectionHeader = typename LP::section;

  auto *hdr =
      reinterpret_cast<const typename LP::mach_header *>(mb.getBufferStart());
  if (hdr->magic != LP::magic)
    return false;

  if (const auto *c =
          findCommand<typename LP::segment_command>(hdr, LP::segmentLCType)) {
    auto sectionHeaders = ArrayRef<SectionHeader>{
        reinterpret_cast<const SectionHeader *>(c + 1), c->nsects};
    for (const SectionHeader &secHead : sectionHeaders) {
      StringRef sectname(secHead.sectname,
                         strnlen(secHead.sectname, sizeof(secHead.sectname)));
      StringRef segname(secHead.segname,
                        strnlen(secHead.segname, sizeof(secHead.segname)));
      if ((segname == segment_names::data &&
           sectname == section_names::objcCatList) ||
          (segname == segment_names::text &&
           sectname.startswith(section_names::swift))) {
        return true;
      }
    }
  }
  return false;
}

static bool objectHasObjCSection(MemoryBufferRef mb) {
  if (target->wordSize == 8)
    return ::objectHasObjCSection<LP64>(mb);
  else
    return ::objectHasObjCSection<ILP32>(mb);
}

bool macho::hasObjCSection(MemoryBufferRef mb) {
  switch (identify_magic(mb.getBuffer())) {
  case file_magic::macho_object:
    return objectHasObjCSection(mb);
  case file_magic::bitcode:
    return check(isBitcodeContainingObjCCategory(mb));
  default:
    return false;
  }
}

namespace {

#define FOR_EACH_CATEGORY_FIELD(DO)                                            \
  DO(Ptr, name)                                                                \
  DO(Ptr, klass)                                                               \
  DO(Ptr, instanceMethods)                                                     \
  DO(Ptr, classMethods)                                                        \
  DO(Ptr, protocols)                                                           \
  DO(Ptr, instanceProps)                                                       \
  DO(Ptr, classProps)

CREATE_LAYOUT_CLASS(Category, FOR_EACH_CATEGORY_FIELD);

#undef FOR_EACH_CATEGORY_FIELD

#define FOR_EACH_CLASS_FIELD(DO)                                               \
  DO(Ptr, metaClass)                                                           \
  DO(Ptr, superClass)                                                          \
  DO(Ptr, methodCache)                                                         \
  DO(Ptr, vtable)                                                              \
  DO(Ptr, roData)

CREATE_LAYOUT_CLASS(Class, FOR_EACH_CLASS_FIELD);

#undef FOR_EACH_CLASS_FIELD

#define FOR_EACH_RO_CLASS_FIELD(DO)                                            \
  DO(uint32_t, flags)                                                          \
  DO(uint32_t, instanceStart)                                                  \
  DO(Ptr, instanceSize)                                                        \
  DO(Ptr, ivarLayout)                                                          \
  DO(Ptr, name)                                                                \
  DO(Ptr, baseMethods)                                                         \
  DO(Ptr, baseProtocols)                                                       \
  DO(Ptr, ivars)                                                               \
  DO(Ptr, weakIvarLayout)                                                      \
  DO(Ptr, baseProperties)

CREATE_LAYOUT_CLASS(ROClass, FOR_EACH_RO_CLASS_FIELD);

#undef FOR_EACH_RO_CLASS_FIELD

#define FOR_EACH_LIST_HEADER(DO)                                               \
  DO(uint32_t, size)                                                           \
  DO(uint32_t, count)

CREATE_LAYOUT_CLASS(ListHeader, FOR_EACH_LIST_HEADER);

#undef FOR_EACH_LIST_HEADER

#define FOR_EACH_METHOD(DO)                                                    \
  DO(Ptr, name)                                                                \
  DO(Ptr, type)                                                                \
  DO(Ptr, impl)

CREATE_LAYOUT_CLASS(Method, FOR_EACH_METHOD);

#undef FOR_EACH_METHOD

enum MethodContainerKind {
  MCK_Class,
  MCK_Category,
};

struct MethodContainer {
  MethodContainerKind kind;
  const ConcatInputSection *isec;
};

enum MethodKind {
  MK_Instance,
  MK_Static,
};

struct ObjcClass {
  DenseMap<CachedHashStringRef, MethodContainer> instanceMethods;
  DenseMap<CachedHashStringRef, MethodContainer> classMethods;
};

} // namespace

class ObjcCategoryChecker {
public:
  ObjcCategoryChecker();
  void parseCategory(const ConcatInputSection *catListIsec);

private:
  void parseClass(const Defined *classSym);
  void parseMethods(const ConcatInputSection *methodsIsec,
                    const Symbol *methodContainer,
                    const ConcatInputSection *containerIsec,
                    MethodContainerKind, MethodKind);

  CategoryLayout catLayout;
  ClassLayout classLayout;
  ROClassLayout roClassLayout;
  ListHeaderLayout listHeaderLayout;
  MethodLayout methodLayout;

  DenseMap<const Symbol *, ObjcClass> classMap;
};

ObjcCategoryChecker::ObjcCategoryChecker()
    : catLayout(target->wordSize), classLayout(target->wordSize),
      roClassLayout(target->wordSize), listHeaderLayout(target->wordSize),
      methodLayout(target->wordSize) {}

// \p r must point to an offset within a cstring section.
static StringRef getReferentString(const Reloc &r) {
  if (auto *isec = r.referent.dyn_cast<InputSection *>())
    return cast<CStringInputSection>(isec)->getStringRefAtOffset(r.addend);
  auto *sym = cast<Defined>(r.referent.get<Symbol *>());
  return cast<CStringInputSection>(sym->isec)->getStringRefAtOffset(sym->value +
                                                                    r.addend);
}

void ObjcCategoryChecker::parseMethods(const ConcatInputSection *methodsIsec,
                                       const Symbol *methodContainerSym,
                                       const ConcatInputSection *containerIsec,
                                       MethodContainerKind mcKind,
                                       MethodKind mKind) {
  ObjcClass &klass = classMap[methodContainerSym];
  for (const Reloc &r : methodsIsec->relocs) {
    if ((r.offset - listHeaderLayout.totalSize) % methodLayout.totalSize !=
        methodLayout.nameOffset)
      continue;

    CachedHashStringRef methodName(getReferentString(r));
    auto &methodMap =
        mKind == MK_Instance ? klass.instanceMethods : klass.classMethods;
    if (methodMap
            .try_emplace(methodName, MethodContainer{mcKind, containerIsec})
            .second)
      continue;

    // We have a duplicate; generate a warning message.
    const auto &mc = methodMap.lookup(methodName);
    const Reloc *nameReloc = nullptr;
    if (mc.kind == MCK_Category) {
      nameReloc = mc.isec->getRelocAt(catLayout.nameOffset);
    } else {
      assert(mc.kind == MCK_Class);
      const auto *roIsec = mc.isec->getRelocAt(classLayout.roDataOffset)
                         ->getReferentInputSection();
      nameReloc = roIsec->getRelocAt(roClassLayout.nameOffset);
    }
    StringRef containerName = getReferentString(*nameReloc);
    StringRef methPrefix = mKind == MK_Instance ? "-" : "+";

    // We should only ever encounter collisions when parsing category methods
    // (since the Class struct is parsed before any of its categories).
    assert(mcKind == MCK_Category);
    StringRef newCatName =
        getReferentString(*containerIsec->getRelocAt(catLayout.nameOffset));

    StringRef containerType = mc.kind == MCK_Category ? "category" : "class";
    warn("method '" + methPrefix + methodName.val() +
         "' has conflicting definitions:\n>>> defined in category " +
         newCatName + " from " + toString(containerIsec->getFile()) +
         "\n>>> defined in " + containerType + " " + containerName + " from " +
         toString(mc.isec->getFile()));
  }
}

void ObjcCategoryChecker::parseCategory(const ConcatInputSection *catIsec) {
  auto *classReloc = catIsec->getRelocAt(catLayout.klassOffset);
  if (!classReloc)
    return;

  auto *classSym = classReloc->referent.get<Symbol *>();
  if (auto *d = dyn_cast<Defined>(classSym))
    if (!classMap.count(d))
      parseClass(d);

  if (const auto *r = catIsec->getRelocAt(catLayout.classMethodsOffset)) {
    parseMethods(cast<ConcatInputSection>(r->getReferentInputSection()),
                 classSym, catIsec, MCK_Category, MK_Static);
  }

  if (const auto *r = catIsec->getRelocAt(catLayout.instanceMethodsOffset)) {
    parseMethods(cast<ConcatInputSection>(r->getReferentInputSection()),
                 classSym, catIsec, MCK_Category, MK_Instance);
  }
}

void ObjcCategoryChecker::parseClass(const Defined *classSym) {
  // Given a Class struct, get its corresponding Methods struct
  auto getMethodsIsec =
      [&](const InputSection *classIsec) -> ConcatInputSection * {
    if (const auto *r = classIsec->getRelocAt(classLayout.roDataOffset)) {
      const auto *roIsec =
          cast<ConcatInputSection>(r->getReferentInputSection());
      if (const auto *r = roIsec->getRelocAt(roClassLayout.baseMethodsOffset)) {
        if (auto *methodsIsec =
                cast_or_null<ConcatInputSection>(r->getReferentInputSection()))
          return methodsIsec;
      }
    }
    return nullptr;
  };

  const auto *classIsec = cast<ConcatInputSection>(classSym->isec);

  // Parse instance methods.
  if (const auto *instanceMethodsIsec = getMethodsIsec(classIsec))
    parseMethods(instanceMethodsIsec, classSym, classIsec, MCK_Class,
                 MK_Instance);

  // Class methods are contained in the metaclass.
  if (const auto *r = classSym->isec->getRelocAt(classLayout.metaClassOffset))
    if (const auto *classMethodsIsec = getMethodsIsec(
            cast<ConcatInputSection>(r->getReferentInputSection())))
      parseMethods(classMethodsIsec, classSym, classIsec, MCK_Class, MK_Static);
}

void objc::checkCategories() {
  ObjcCategoryChecker checker;
  for (const InputSection *isec : inputSections) {
    if (isec->getName() == section_names::objcCatList)
      for (const Reloc &r : isec->relocs) {
        auto *catIsec = cast<ConcatInputSection>(r.getReferentInputSection());
        checker.parseCategory(catIsec);
      }
  }
}
