//===- ObjC.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjC.h"
#include "ConcatOutputSection.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "Layout.h"
#include "OutputSegment.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/TimeProfiler.h"

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
           sectname.starts_with(section_names::swift))) {
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
  DO(Ptr, classProps)                                                          \
  DO(uint32_t, size)

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
  DO(uint32_t, structSize)                                                     \
  DO(uint32_t, structCount)

CREATE_LAYOUT_CLASS(ListHeader, FOR_EACH_LIST_HEADER);

#undef FOR_EACH_LIST_HEADER

#define FOR_EACH_PROTOCOL_LIST_HEADER(DO) DO(Ptr, protocolCount)

CREATE_LAYOUT_CLASS(ProtocolListHeader, FOR_EACH_PROTOCOL_LIST_HEADER);

#undef FOR_EACH_PROTOCOL_LIST_HEADER

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
    // +load methods are special: all implementations are called by the runtime
    // even if they are part of the same class. Thus there is no need to check
    // for duplicates.
    // NOTE: Instead of specifically checking for this method name, ld64 simply
    // checks whether a class / category is present in __objc_nlclslist /
    // __objc_nlcatlist respectively. This will be the case if the class /
    // category has a +load method. It skips optimizing the categories if there
    // are multiple +load methods. Since it does dupe checking as part of the
    // optimization process, this avoids spurious dupe messages around +load,
    // but it also means that legit dupe issues for other methods are ignored.
    if (mKind == MK_Static && methodName.val() == "load")
      continue;

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

    auto formatObjAndSrcFileName = [](const InputSection *section) {
      lld::macho::InputFile *inputFile = section->getFile();
      std::string result = toString(inputFile);

      auto objFile = dyn_cast_or_null<ObjFile>(inputFile);
      if (objFile && objFile->compileUnit)
        result += " (" + objFile->sourceFile() + ")";

      return result;
    };

    StringRef containerType = mc.kind == MCK_Category ? "category" : "class";
    warn("method '" + methPrefix + methodName.val() +
         "' has conflicting definitions:\n>>> defined in category " +
         newCatName + " from " + formatObjAndSrcFileName(containerIsec) +
         "\n>>> defined in " + containerType + " " + containerName + " from " +
         formatObjAndSrcFileName(mc.isec));
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
      if (const auto *roIsec =
              cast_or_null<ConcatInputSection>(r->getReferentInputSection())) {
        if (const auto *r =
                roIsec->getRelocAt(roClassLayout.baseMethodsOffset)) {
          if (auto *methodsIsec = cast_or_null<ConcatInputSection>(
                  r->getReferentInputSection()))
            return methodsIsec;
        }
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
  TimeTraceScope timeScope("ObjcCategoryChecker");

  ObjcCategoryChecker checker;
  for (const InputSection *isec : inputSections) {
    if (isec->getName() == section_names::objcCatList)
      for (const Reloc &r : isec->relocs) {
        auto *catIsec = cast<ConcatInputSection>(r.getReferentInputSection());
        checker.parseCategory(catIsec);
      }
  }
}

namespace {

class ObjcCategoryMerger {
  // Information about an input category
  struct InfoInputCategory {
    ConcatInputSection *catBodyIsec;
    ConcatInputSection *catListIsec;
    uint32_t offCatListIsec = 0;

    bool wasMerged = false;
  };

  // To write new (merged) categories or classes, we will try make limited
  // assumptions about the alignment and the sections the various class/category
  // info are stored in and . So we'll just reuse the same sections and
  // alignment as already used in existing (input) categories. To do this we
  // have InfoCategoryWriter which contains the various sections that the
  // generated categories will be written to.
  template <typename T> struct InfroWriteSection {
    bool valid = false; // Data has been successfully collected from input
    uint32_t align = 0;
    Section *inputSection;
    Reloc relocTemplate;
    T *outputSection;
  };

  struct InfoCategoryWriter {
    InfroWriteSection<ConcatOutputSection> catListInfo;
    InfroWriteSection<ConcatOutputSection> catBodyInfo;
    InfroWriteSection<CStringSection> catNameInfo;
    InfroWriteSection<ConcatOutputSection> catPtrListInfo;
  };

  // Information about a pointer list in the original categories (method lists,
  // protocol lists, etc)
  struct PointerListInfo {
    PointerListInfo(const char *pszSymNamePrefix)
        : namePrefix(pszSymNamePrefix) {}
    const char *namePrefix;

    uint32_t structSize = 0;
    uint32_t structCount = 0;

    std::vector<Symbol *> allPtrs;
  };

  // Full information about all the categories that are extending a class. This
  // will have all the additional methods, protocols, proprieties that are
  // contained in all the categories that extend a particular class.
  struct ClassExtensionInfo {
    // Merged names of containers. Ex: base|firstCategory|secondCategory|...
    std::string mergedContainerName;
    std::string baseClassName;
    Symbol *baseClass = nullptr;
    // In case we generate new data, mark the new data as belonging to this file
    ObjFile *objFileForMergeData = nullptr;

    PointerListInfo instanceMethods =
        objc::symbol_names::categoryInstanceMethods;
    PointerListInfo classMethods = objc::symbol_names::categoryClassMethods;
    PointerListInfo protocols = objc::symbol_names::categoryProtocols;
    PointerListInfo instanceProps = objc::symbol_names::listProprieties;
    PointerListInfo classProps = objc::symbol_names::klassPropList;
  };

public:
  ObjcCategoryMerger(std::vector<ConcatInputSection *> &_allInputSections);
  void doMerge();
  static void doCleanup();

private:
  void collectAndValidateCategoriesData();
  void
  mergeCategoriesIntoSingleCategory(std::vector<InfoInputCategory> &categories);

  void eraseISec(ConcatInputSection *isec);
  void eraseMergedCategories();

  void generateCatListForNonErasedCategories(
      std::map<ConcatInputSection *, std::set<uint64_t>>
          catListToErasedOffsets);
  template <typename T>
  void collectSectionWriteInfoFromIsec(InputSection *isec,
                                       InfroWriteSection<T> &catWriteInfo);
  void collectCategoryWriterInfoFromCategory(InfoInputCategory &catInfo);
  void parseCatInfoToExtInfo(InfoInputCategory &catInfo,
                             ClassExtensionInfo &extInfo);

  void parseProtocolListInfo(ConcatInputSection *isec,
                             uint32_t symbolsPerStruct,
                             PointerListInfo &ptrList);

  void parsePointerListInfo(ConcatInputSection *isec, uint32_t secOffset,
                            uint32_t symbolsPerStruct,
                            PointerListInfo &ptrList);

  void emitAndLinkPointerList(Defined *parentSym, uint32_t linkAtOffset,
                              ClassExtensionInfo &extInfo,
                              PointerListInfo &ptrList);

  void emitAndLinkProtocolList(Defined *parentSym, uint32_t linkAtOffset,
                               ClassExtensionInfo &extInfo,
                               PointerListInfo &ptrList);

  void emitCategory(ClassExtensionInfo &extInfo, Defined *&catBodySym);
  void emitCatListEntrySec(std::string &forCateogryName,
                           std::string &forBaseClassName, ObjFile *objFile,
                           Defined *&catListSym);
  void emitCategoryBody(std::string &name, Defined *nameSym,
                        Symbol *baseClassSym, std::string &baseClassName,
                        ObjFile *objFile, Defined *&catBodySym);
  void emitCategoryName(std::string &name, ObjFile *objFile,
                        Defined *&catNameSym);
  void createSymbolReference(Defined *refFrom, Symbol *refTo, uint32_t offset,
                             Reloc &relocTemplate);
  bool tryGetSymbolAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
                                Symbol *&sym);
  bool tryGetDefinedAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
                                 Defined *&defined);
  void tryEraseDefinedAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
                                   bool stringOnly = false);

  CategoryLayout catLayout;
  ClassLayout classLayout;
  ROClassLayout roClassLayout;
  ListHeaderLayout listHeaderLayout;
  MethodLayout methodLayout;
  ProtocolListHeaderLayout protocolListHeaderLayout;

  InfoCategoryWriter infoCategoryWriter;
  std::vector<ConcatInputSection *> &allInputSections;
  // Map of base class Symbol to list of InfoInputCategory's for it
  std::map<const Symbol *, std::vector<InfoInputCategory>> categoryMap;

  // Normally, the binary data comes from the input files, but since we're
  // generating binary data ourselves, we use the below arrays to store it in.
  // Need this to be 'static' so the data survives past the ObjcCategoryMerger
  // object, as the data will be read by the Writer when the final binary is
  // generated.
  static SmallVector<SmallString<0>> generatedNames;
  static SmallVector<SmallVector<uint8_t>> generatedSectionData;
};

SmallVector<SmallString<0>> ObjcCategoryMerger::generatedNames;
SmallVector<SmallVector<uint8_t>> ObjcCategoryMerger::generatedSectionData;

ObjcCategoryMerger::ObjcCategoryMerger(
    std::vector<ConcatInputSection *> &_allInputSections)
    : catLayout(target->wordSize), classLayout(target->wordSize),
      roClassLayout(target->wordSize), listHeaderLayout(target->wordSize),
      methodLayout(target->wordSize),
      protocolListHeaderLayout(target->wordSize),
      allInputSections(_allInputSections) {}

// This is a template so that it can be used both for CStringSection and
// ConcatOutputSection
template <typename T>
void ObjcCategoryMerger::collectSectionWriteInfoFromIsec(
    InputSection *isec, InfroWriteSection<T> &catWriteInfo) {

  catWriteInfo.inputSection = const_cast<Section *>(&isec->section);
  catWriteInfo.align = isec->align;
  catWriteInfo.outputSection = dyn_cast_or_null<T>(isec->parent);

  assert(catWriteInfo.outputSection &&
         "outputSection may not be null in collectSectionWriteInfoFromIsec.");

  if (isec->relocs.size())
    catWriteInfo.relocTemplate = isec->relocs[0];

  catWriteInfo.valid = true;
}

bool ObjcCategoryMerger::tryGetSymbolAtIsecOffset(ConcatInputSection *isec,
                                                  uint32_t offset,
                                                  Symbol *&sym) {
  const Reloc *reloc = isec->getRelocAt(offset);

  if (!reloc)
    return false;

  sym = reloc->referent.get<Symbol *>();
  return sym != nullptr;
}

bool ObjcCategoryMerger::tryGetDefinedAtIsecOffset(ConcatInputSection *isec,
                                                   uint32_t offset,
                                                   Defined *&defined) {
  Symbol *sym;
  if (!tryGetSymbolAtIsecOffset(isec, offset, sym))
    return false;

  defined = dyn_cast_or_null<Defined>(sym);
  return defined != nullptr;
}

// Given an ConcatInputSection and an offset, if there is a symbol(Defined) at
// that offset, then erase the symbol (mark it not live) from the final output.
// Used for easely erasing already merged strings, method lists, etc ...
void ObjcCategoryMerger::tryEraseDefinedAtIsecOffset(ConcatInputSection *isec,
                                                     uint32_t offset,
                                                     bool stringOnly) {
  const Reloc *reloc = isec->getRelocAt(offset);

  if (!reloc)
    return;

  Defined *sym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
  if (!sym)
    return;

  auto *cisec = dyn_cast_or_null<ConcatInputSection>(sym->isec);
  if (!stringOnly && cisec) {
    eraseISec(cisec);
    return;
  }

  if (auto *cisec = dyn_cast_or_null<CStringInputSection>(sym->isec)) {
    uint32_t totalOffset = sym->value + reloc->addend;
    StringPiece &piece = cisec->getStringPiece(totalOffset);
    piece.live = false;
    return;
  }
}

void ObjcCategoryMerger::collectCategoryWriterInfoFromCategory(
    InfoInputCategory &catInfo) {

  collectSectionWriteInfoFromIsec<ConcatOutputSection>(
      catInfo.catListIsec, infoCategoryWriter.catListInfo);
  collectSectionWriteInfoFromIsec<ConcatOutputSection>(
      catInfo.catBodyIsec, infoCategoryWriter.catBodyInfo);

  if (!infoCategoryWriter.catNameInfo.valid) {
    const Reloc *catNameReloc =
        catInfo.catBodyIsec->getRelocAt(catLayout.nameOffset);

    assert(catNameReloc && "Category does not have a reloc at nameOffset");

    lld::macho::Defined *catDefSym =
        dyn_cast_or_null<Defined>(catNameReloc->referent.dyn_cast<Symbol *>());
    assert(catDefSym && "Reloc of category name is not a valid Defined symbol");

    collectSectionWriteInfoFromIsec<CStringSection>(
        catDefSym->isec, infoCategoryWriter.catNameInfo);
  }

  // Collect writer info from all the category lists (we're assuming they all
  // would provide the same info)
  if (!infoCategoryWriter.catPtrListInfo.valid) {
    for (uint32_t off = catLayout.instanceMethodsOffset;
         off <= catLayout.classPropsOffset; off += target->wordSize) {
      Defined *ptrList;
      if (tryGetDefinedAtIsecOffset(catInfo.catBodyIsec, off, ptrList)) {
        collectSectionWriteInfoFromIsec<ConcatOutputSection>(
            ptrList->isec, infoCategoryWriter.catPtrListInfo);
        // we've successfully collected data, so we can break
        break;
      }
    }
  }
}

// Parse a protocol list that might be linked to at a ConcatInputSection given
// offset. The format of the protocol list is different than other lists (prop
// lists, method lists) so we need to parse it differently
void ObjcCategoryMerger::parseProtocolListInfo(ConcatInputSection *isec,
                                               uint32_t secOffset,
                                               PointerListInfo &ptrList) {
  if (!isec || (secOffset + target->wordSize > isec->data.size()))
    assert("Tried to read pointer list beyond protocol section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  if (!reloc)
    return; // List is null, nothing to do

  auto *ptrListSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
  assert(ptrListSym && "Protocol list reloc does not have a valid Defined");

  // Theoretically protocol count can be either 32b or 64b, but reading the
  // first 32b is good enough
  uint32_t protocolCount = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec->data.data() + listHeaderLayout.structSizeOffset);

  ptrList.structCount += protocolCount;
  ptrList.structSize = target->wordSize;

  uint32_t expectedListSize =
      (protocolCount * target->wordSize) +
      /*header(count)*/ protocolListHeaderLayout.totalSize +
      /*extra null value*/ target->wordSize;
  assert(expectedListSize == ptrListSym->isec->data.size() &&
         "Protocol list does not match expected size");

  uint32_t off = protocolListHeaderLayout.totalSize;
  for (uint32_t inx = 0; inx < protocolCount; inx++) {
    const Reloc *reloc = ptrListSym->isec->getRelocAt(off);
    assert(reloc && "No reloc found at protocol list offset");

    auto *listSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
    assert(listSym && "Protocol list reloc does not have a valid Defined");

    ptrList.allPtrs.push_back(listSym);
    off += target->wordSize;
  }
}

// Parse a pointer list that might be linked to at a ConcatInputSection given
// offset. This can be used for instance methods, class methods, instance props
// and class props since they have the same format.
void ObjcCategoryMerger::parsePointerListInfo(ConcatInputSection *isec,
                                              uint32_t secOffset,
                                              uint32_t symbolsPerStruct,
                                              PointerListInfo &ptrList) {
  assert(symbolsPerStruct == 2 || symbolsPerStruct == 3);
  assert(isec && "Trying to parse pointer list from null isec");
  assert(secOffset + target->wordSize <= isec->data.size() &&
         "Trying to read pointer list beyond section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  if (!reloc)
    return; // No reloc found, nothing to parse

  auto *ptrListSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
  assert(ptrListSym && "Reloc does not have a valid Defined");

  uint32_t thisStructSize = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec->data.data() + listHeaderLayout.structSizeOffset);
  uint32_t thisStructCount = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec->data.data() + listHeaderLayout.structCountOffset);

  assert(!ptrList.structSize || (thisStructSize == ptrList.structSize));

  ptrList.structCount += thisStructCount;
  ptrList.structSize = thisStructSize;

  uint32_t expectedListSize =
      listHeaderLayout.totalSize + (thisStructSize * thisStructCount);

  assert(expectedListSize == ptrListSym->isec->data.size() &&
         "Pointer list does not match expected size");

  for (uint32_t off = listHeaderLayout.totalSize; off < expectedListSize;
       off += target->wordSize) {
    const Reloc *reloc = ptrListSym->isec->getRelocAt(off);
    assert(reloc && "No reloc found at pointer list offset");

    auto *listSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
    assert(listSym && "Reloc does not have a valid Defined");

    ptrList.allPtrs.push_back(listSym);
  }
}

// Here we parse all the information of an input category (catInfo) and
// append-store the parsed info into the strucutre which will contain all the
// information about how a class is extended (extInfo)
void ObjcCategoryMerger::parseCatInfoToExtInfo(InfoInputCategory &catInfo,
                                               ClassExtensionInfo &extInfo) {
  const Reloc *catNameReloc =
      catInfo.catBodyIsec->getRelocAt(catLayout.nameOffset);

  // Parse name
  assert(catNameReloc && "Category does not have a reloc at 'nameOffset'");

  if (!extInfo.mergedContainerName.empty())
    extInfo.mergedContainerName += "|";

  if (!extInfo.objFileForMergeData)
    extInfo.objFileForMergeData =
        dyn_cast_or_null<ObjFile>(catInfo.catBodyIsec->getFile());

  StringRef catName = getReferentString(*catNameReloc);
  extInfo.mergedContainerName += catName.str();

  // Parse base class
  const Reloc *klassReloc =
      catInfo.catBodyIsec->getRelocAt(catLayout.klassOffset);

  assert(klassReloc && "Category does not have a reloc at 'klassOffset'");

  Symbol *classSym = klassReloc->referent.get<Symbol *>();

  assert(
      (!extInfo.baseClass || (extInfo.baseClass == classSym)) &&
      "Trying to parse category info into container with different base class");

  extInfo.baseClass = classSym;

  if (extInfo.baseClassName.empty()) {
    llvm::StringRef classPrefix(objc::symbol_names::klass);
    assert(classSym->getName().starts_with(classPrefix) &&
           "Base class symbol does not start with expected prefix");

    extInfo.baseClassName = classSym->getName().substr(classPrefix.size());
  }

  parsePointerListInfo(catInfo.catBodyIsec, catLayout.instanceMethodsOffset,
                       /*symbolsPerStruct=*/3, extInfo.instanceMethods);

  parsePointerListInfo(catInfo.catBodyIsec, catLayout.classMethodsOffset,
                       /*symbolsPerStruct=*/3, extInfo.classMethods);

  parseProtocolListInfo(catInfo.catBodyIsec, catLayout.protocolsOffset,
                        extInfo.protocols);

  parsePointerListInfo(catInfo.catBodyIsec, catLayout.instancePropsOffset,
                       /*symbolsPerStruct=*/2, extInfo.instanceProps);

  parsePointerListInfo(catInfo.catBodyIsec, catLayout.classPropsOffset,
                       /*symbolsPerStruct=*/2, extInfo.classProps);
}

// Generate a protocol list (including header) and link it into the parent at
// the specified offset.
void ObjcCategoryMerger::emitAndLinkProtocolList(Defined *parentSym,
                                                 uint32_t linkAtOffset,
                                                 ClassExtensionInfo &extInfo,
                                                 PointerListInfo &ptrList) {
  if (ptrList.allPtrs.empty())
    return;

  assert(ptrList.allPtrs.size() == ptrList.structCount);

  uint32_t bodySize = (ptrList.structCount * target->wordSize) +
                      /*header(count)*/ protocolListHeaderLayout.totalSize +
                      /*extra null value*/ target->wordSize;
  generatedSectionData.push_back(SmallVector<uint8_t>(bodySize, 0));
  llvm::ArrayRef<uint8_t> bodyData = generatedSectionData.back();

  // This theoretically can be either 32b or 64b, but writing just the first 32b
  // is good enough
  const uint32_t *ptrProtoCount = reinterpret_cast<const uint32_t *>(
      bodyData.data() + protocolListHeaderLayout.protocolCountOffset);

  *const_cast<uint32_t *>(ptrProtoCount) = ptrList.allPtrs.size();

  ConcatInputSection *listSec = make<ConcatInputSection>(
      *infoCategoryWriter.catPtrListInfo.inputSection, bodyData,
      infoCategoryWriter.catPtrListInfo.align);
  listSec->parent = infoCategoryWriter.catPtrListInfo.outputSection;
  listSec->live = true;
  allInputSections.push_back(listSec);

  listSec->parent = infoCategoryWriter.catPtrListInfo.outputSection;

  generatedNames.push_back(StringRef(ptrList.namePrefix));
  auto &symName = generatedNames.back();
  symName += extInfo.baseClassName + "_$_(" + extInfo.mergedContainerName + ")";

  Defined *ptrListSym = make<Defined>(
      symName.c_str(), /*file=*/parentSym->getObjectFile(), listSec,
      /*value=*/0, bodyData.size(),
      /*isWeakDef=*/false, /*isExternal=*/false, /*isPrivateExtern=*/false,
      /*includeInSymtab=*/true, /*isReferencedDynamically=*/false,
      /*noDeadStrip=*/false, /*isWeakDefCanBeHidden=*/false);

  ptrListSym->used = true;
  parentSym->getObjectFile()->symbols.push_back(ptrListSym);

  createSymbolReference(parentSym, ptrListSym, linkAtOffset,
                        infoCategoryWriter.catBodyInfo.relocTemplate);

  uint32_t offset = protocolListHeaderLayout.totalSize;
  for (Symbol *symbol : ptrList.allPtrs) {
    createSymbolReference(ptrListSym, symbol, offset,
                          infoCategoryWriter.catPtrListInfo.relocTemplate);
    offset += target->wordSize;
  }
}

// Generate a pointer list (including header) and link it into the parent at the
// specified offset. This is used for instance and class methods and
// proprieties.
void ObjcCategoryMerger::emitAndLinkPointerList(Defined *parentSym,
                                                uint32_t linkAtOffset,
                                                ClassExtensionInfo &extInfo,
                                                PointerListInfo &ptrList) {
  if (ptrList.allPtrs.empty())
    return;

  assert(ptrList.allPtrs.size() * target->wordSize ==
         ptrList.structCount * ptrList.structSize);

  // Generate body
  uint32_t bodySize =
      listHeaderLayout.totalSize + (ptrList.structSize * ptrList.structCount);
  generatedSectionData.push_back(SmallVector<uint8_t>(bodySize, 0));
  llvm::ArrayRef<uint8_t> bodyData = generatedSectionData.back();

  const uint32_t *ptrStructSize = reinterpret_cast<const uint32_t *>(
      bodyData.data() + listHeaderLayout.structSizeOffset);
  const uint32_t *ptrStructCount = reinterpret_cast<const uint32_t *>(
      bodyData.data() + listHeaderLayout.structCountOffset);

  *const_cast<uint32_t *>(ptrStructSize) = ptrList.structSize;
  *const_cast<uint32_t *>(ptrStructCount) = ptrList.structCount;

  ConcatInputSection *listSec = make<ConcatInputSection>(
      *infoCategoryWriter.catPtrListInfo.inputSection, bodyData,
      infoCategoryWriter.catPtrListInfo.align);
  listSec->parent = infoCategoryWriter.catPtrListInfo.outputSection;
  listSec->live = true;
  allInputSections.push_back(listSec);

  listSec->parent = infoCategoryWriter.catPtrListInfo.outputSection;

  generatedNames.push_back(StringRef(ptrList.namePrefix));
  auto &symName = generatedNames.back();
  symName += extInfo.baseClassName + "_$_" + extInfo.mergedContainerName;

  Defined *ptrListSym = make<Defined>(
      symName.c_str(), /*file=*/parentSym->getObjectFile(), listSec,
      /*value=*/0, bodyData.size(),
      /*isWeakDef=*/false, /*isExternal=*/false, /*isPrivateExtern=*/false,
      /*includeInSymtab=*/true, /*isReferencedDynamically=*/false,
      /*noDeadStrip=*/false, /*isWeakDefCanBeHidden=*/false);

  ptrListSym->used = true;
  parentSym->getObjectFile()->symbols.push_back(ptrListSym);

  createSymbolReference(parentSym, ptrListSym, linkAtOffset,
                        infoCategoryWriter.catBodyInfo.relocTemplate);

  uint32_t offset = listHeaderLayout.totalSize;
  for (Symbol *symbol : ptrList.allPtrs) {
    createSymbolReference(ptrListSym, symbol, offset,
                          infoCategoryWriter.catPtrListInfo.relocTemplate);
    offset += target->wordSize;
  }
}

// This method creates an __objc_catlist ConcatInputSection with a single slot
void ObjcCategoryMerger::emitCatListEntrySec(std::string &forCateogryName,
                                             std::string &forBaseClassName,
                                             ObjFile *objFile,
                                             Defined *&catListSym) {
  uint32_t sectionSize = target->wordSize;
  generatedSectionData.push_back(SmallVector<uint8_t>(sectionSize, 0));
  llvm::ArrayRef<uint8_t> bodyData = generatedSectionData.back();

  ConcatInputSection *newCatList =
      make<ConcatInputSection>(*infoCategoryWriter.catListInfo.inputSection,
                               bodyData, infoCategoryWriter.catListInfo.align);
  newCatList->parent = infoCategoryWriter.catListInfo.outputSection;
  newCatList->live = true;
  allInputSections.push_back(newCatList);

  newCatList->parent = infoCategoryWriter.catListInfo.outputSection;

  SmallString<0> catSymName;
  catSymName += "<__objc_catlist slot for merged category ";
  catSymName += forBaseClassName + "(" + forCateogryName + ")>";
  generatedNames.push_back(StringRef(catSymName));

  catListSym = make<Defined>(
      StringRef(generatedNames.back()), /*file=*/objFile, newCatList,
      /*value=*/0, bodyData.size(), /*isWeakDef=*/false, /*isExternal=*/false,
      /*isPrivateExtern=*/false, /*includeInSymtab=*/false,
      /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
      /*isWeakDefCanBeHidden=*/false);

  catListSym->used = true;
  objFile->symbols.push_back(catListSym);
}

// Here we generate the main category body and just the body and link the name
// and base class into it. We don't link any other info like the protocol and
// class/instance methods/props.
void ObjcCategoryMerger::emitCategoryBody(std::string &name, Defined *nameSym,
                                          Symbol *baseClassSym,
                                          std::string &baseClassName,
                                          ObjFile *objFile,
                                          Defined *&catBodySym) {
  generatedSectionData.push_back(SmallVector<uint8_t>(catLayout.totalSize, 0));
  llvm::ArrayRef<uint8_t> bodyData = generatedSectionData.back();

  uint32_t *ptrSize = (uint32_t *)(const_cast<uint8_t *>(bodyData.data()) +
                                   catLayout.sizeOffset);
  *ptrSize = catLayout.totalSize;

  ConcatInputSection *newBodySec =
      make<ConcatInputSection>(*infoCategoryWriter.catBodyInfo.inputSection,
                               bodyData, infoCategoryWriter.catBodyInfo.align);
  newBodySec->parent = infoCategoryWriter.catBodyInfo.outputSection;
  newBodySec->live = true;
  allInputSections.push_back(newBodySec);

  newBodySec->parent = infoCategoryWriter.catBodyInfo.outputSection;

  std::string symName =
      objc::symbol_names::category + baseClassName + "_$_(" + name + ")";
  generatedNames.push_back(StringRef(symName));
  catBodySym = make<Defined>(
      StringRef(generatedNames.back()), /*file=*/objFile, newBodySec,
      /*value=*/0, bodyData.size(), /*isWeakDef=*/false, /*isExternal=*/false,
      /*isPrivateExtern=*/false, /*includeInSymtab=*/true,
      /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
      /*isWeakDefCanBeHidden=*/false);

  catBodySym->used = true;
  objFile->symbols.push_back(catBodySym);

  createSymbolReference(catBodySym, nameSym, catLayout.nameOffset,
                        infoCategoryWriter.catBodyInfo.relocTemplate);

  // Create a reloc to the base class (either external or internal)
  createSymbolReference(catBodySym, baseClassSym, catLayout.klassOffset,
                        infoCategoryWriter.catBodyInfo.relocTemplate);
}

// This writes the new category name (for the merged category) into the binary
// and returns the sybmol for it.
void ObjcCategoryMerger::emitCategoryName(std::string &name, ObjFile *objFile,
                                          Defined *&catNamdeSym) {
  llvm::ArrayRef<uint8_t> inputNameArrData(
      reinterpret_cast<const uint8_t *>(name.c_str()), name.size() + 1);
  generatedSectionData.push_back(SmallVector<uint8_t>(inputNameArrData));

  llvm::ArrayRef<uint8_t> nameData = generatedSectionData.back();

  auto *parentSection = infoCategoryWriter.catNameInfo.inputSection;
  CStringInputSection *newStringSec = make<CStringInputSection>(
      *infoCategoryWriter.catNameInfo.inputSection, nameData,
      infoCategoryWriter.catNameInfo.align, true);

  parentSection->subsections.push_back({0, newStringSec});

  newStringSec->splitIntoPieces();
  newStringSec->pieces[0].live = true;
  newStringSec->parent = infoCategoryWriter.catNameInfo.outputSection;

  catNamdeSym = make<Defined>(
      "<merged category name>", /*file=*/objFile, newStringSec,
      /*value=*/0, nameData.size(),
      /*isWeakDef=*/false, /*isExternal=*/false, /*isPrivateExtern=*/false,
      /*includeInSymtab=*/false, /*isReferencedDynamically=*/false,
      /*noDeadStrip=*/false, /*isWeakDefCanBeHidden=*/false);

  catNamdeSym->used = true;
  objFile->symbols.push_back(catNamdeSym);
}

// This method fully creates a new category from the given ClassExtensionInfo.
// It creates the category body, name and protocol/method/prop lists an links
// everything together. Then it creates a new __objc_catlist entry and links the
// category into it. Calling this method will fully generate a category which
// will be available in the final binary.
void ObjcCategoryMerger::emitCategory(ClassExtensionInfo &extInfo,
                                      Defined *&catBodySym) {
  Defined *catNameSym = nullptr;
  emitCategoryName(extInfo.mergedContainerName, extInfo.objFileForMergeData,
                   catNameSym);

  emitCategoryBody(extInfo.mergedContainerName, catNameSym, extInfo.baseClass,
                   extInfo.baseClassName, extInfo.objFileForMergeData,
                   catBodySym);

  Defined *catListSym = nullptr;
  emitCatListEntrySec(extInfo.mergedContainerName, extInfo.baseClassName,
                      extInfo.objFileForMergeData, catListSym);

  const uint32_t offsetFirstCat = 0;
  createSymbolReference(catListSym, catBodySym, offsetFirstCat,
                        infoCategoryWriter.catListInfo.relocTemplate);

  emitAndLinkPointerList(catBodySym, catLayout.instanceMethodsOffset, extInfo,
                         extInfo.instanceMethods);

  emitAndLinkPointerList(catBodySym, catLayout.classMethodsOffset, extInfo,
                         extInfo.classMethods);

  emitAndLinkProtocolList(catBodySym, catLayout.protocolsOffset, extInfo,
                          extInfo.protocols);

  emitAndLinkPointerList(catBodySym, catLayout.instancePropsOffset, extInfo,
                         extInfo.instanceProps);

  emitAndLinkPointerList(catBodySym, catLayout.classPropsOffset, extInfo,
                         extInfo.classProps);
}

// This method merges all the categories (sharing a base class) into a single
// category.
void ObjcCategoryMerger::mergeCategoriesIntoSingleCategory(
    std::vector<InfoInputCategory> &categories) {
  assert(categories.size() > 1 && "Expected at least 2 categories");

  ClassExtensionInfo extInfo;

  for (auto &catInfo : categories)
    parseCatInfoToExtInfo(catInfo, extInfo);

  Defined *newCatDef = nullptr;
  emitCategory(extInfo, newCatDef);
  assert(newCatDef && "Failed to create a new category");
}

void ObjcCategoryMerger::createSymbolReference(Defined *refFrom, Symbol *refTo,
                                               uint32_t offset,
                                               Reloc &relocTemplate) {
  Reloc r = relocTemplate;
  r.offset = offset;
  r.addend = 0;
  r.referent = refTo;
  refFrom->isec->relocs.push_back(r);
}

void ObjcCategoryMerger::collectAndValidateCategoriesData() {
  for (InputSection *sec : allInputSections) {
    if (sec->getName() != section_names::objcCatList)
      continue;
    ConcatInputSection *catListCisec = dyn_cast<ConcatInputSection>(sec);
    assert(catListCisec &&
           "__objc_catList InputSection is not a ConcatInputSection");

    for (uint32_t off = 0; off < catListCisec->getSize();
         off += target->wordSize) {
      Defined *categorySym = nullptr;
      tryGetDefinedAtIsecOffset(catListCisec, off, categorySym);
      assert(categorySym &&
             "Failed to get a valid cateogry at __objc_catlit offset");
      if (!categorySym->getName().starts_with(objc::symbol_names::category))
        continue; // Only support ObjC categories (no swift + @objc)

      auto *catBodyIsec = dyn_cast<ConcatInputSection>(categorySym->isec);
      assert(catBodyIsec &&
             "Category data section is not an ConcatInputSection");

      // Check that the category has a reloc at 'klassOffset' (which is
      // a pointer to the class symbol)

      Symbol *classSym = nullptr;
      tryGetSymbolAtIsecOffset(catBodyIsec, catLayout.klassOffset, classSym);
      assert(classSym && "Category does not have a valid base class");

      InfoInputCategory catInputInfo{catBodyIsec, catListCisec, off};
      categoryMap[classSym].push_back(catInputInfo);

      collectCategoryWriterInfoFromCategory(catInputInfo);
    }
  }

  for (auto &entry : categoryMap) {
    if (entry.second.size() > 1) {
      // Sort categories by offset to make sure we process categories in
      // the same order as they appear in the input
      auto cmpFn = [](const InfoInputCategory &a, const InfoInputCategory &b) {
        return (a.catListIsec == b.catListIsec) &&
               (a.offCatListIsec < b.offCatListIsec);
      };

      llvm::sort(entry.second, cmpFn);
    }
  }
}

// In the input we have multiple __objc_catlist InputSection, each of which may
// contain links to multiple categories. Of these categories, we will merge (and
// erase) only some. There will be some categories that will remain unoutched
// (not erased). For these not erased categories, we generate new __objc_catlist
// entries since the parent __objc_catlist entry will be erased
void ObjcCategoryMerger::generateCatListForNonErasedCategories(
    std::map<ConcatInputSection *, std::set<uint64_t>> catListToErasedOffsets) {

  // Go through all offsets of all __objc_catlist's that we process and if there
  // are categories that we didn't process - generate a new __objv_catlist for
  // each.
  for (auto &mapEntry : catListToErasedOffsets) {
    ConcatInputSection *catListIsec = mapEntry.first;
    uint32_t catListIsecOffset = 0;
    while (catListIsecOffset < catListIsec->data.size()) {
      // This slot was erased, we can jsut skip it
      if (mapEntry.second.count(catListIsecOffset)) {
        catListIsecOffset += target->wordSize;
        continue;
      }

      Defined *nonErasedCatBody = nullptr;
      tryGetDefinedAtIsecOffset(catListIsec, catListIsecOffset,
                                nonErasedCatBody);
      assert(nonErasedCatBody && "Failed to relocate non-deleted category");

      // Allocate data for the new __objc_catlist slot
      generatedSectionData.push_back(SmallVector<uint8_t>(target->wordSize, 0));
      llvm::ArrayRef<uint8_t> bodyData = generatedSectionData.back();

      // We mark the __objc_catlist slot as belonging to the same file as the
      // category
      ObjFile *objFile = dyn_cast<ObjFile>(nonErasedCatBody->getFile());

      ConcatInputSection *listSec = make<ConcatInputSection>(
          *infoCategoryWriter.catListInfo.inputSection, bodyData,
          infoCategoryWriter.catListInfo.align);
      listSec->parent = infoCategoryWriter.catListInfo.outputSection;
      listSec->live = true;
      allInputSections.push_back(listSec);

      generatedNames.push_back(StringRef("<__objc_catlist slot for category "));
      auto &slotSymName = generatedNames.back();
      slotSymName += nonErasedCatBody->getName();
      slotSymName += ">";

      Defined *catListSlotSym = make<Defined>(
          slotSymName.c_str(), /*file=*/objFile, listSec,
          /*value=*/0, bodyData.size(),
          /*isWeakDef=*/false, /*isExternal=*/false, /*isPrivateExtern=*/false,
          /*includeInSymtab=*/false, /*isReferencedDynamically=*/false,
          /*noDeadStrip=*/false, /*isWeakDefCanBeHidden=*/false);

      catListSlotSym->used = true;
      objFile->symbols.push_back(catListSlotSym);

      // Now link the category body into the newly created slot
      createSymbolReference(catListSlotSym, nonErasedCatBody, 0,
                            infoCategoryWriter.catListInfo.relocTemplate);

      catListIsecOffset += target->wordSize;
    }
  }
}

void ObjcCategoryMerger::eraseISec(ConcatInputSection *isec) {
  isec->live = false;
  for (auto &sym : isec->symbols) {
    sym->used = false;
  }
}

// This fully erases the merged categories, including their body, their names,
// their method/protocol/prop lists and the __objc_catlist entries that link to
// them.
void ObjcCategoryMerger::eraseMergedCategories() {
  // We expect there to be many categories in an input __objc_catList, so we
  // can't just, of which we will merge only some. Because of this, we can't
  // just erase the entire __objc_catList, we need to erase the merged
  // categories only. To do this, we generate a new __objc_catList and copy over
  // all the un-merged categories and erase all the affected (and only the
  // affected) __objc_catList's

  // Map of InputSection to a set of offsets of the categories that were merged
  std::map<ConcatInputSection *, std::set<uint64_t>> catListToErasedOffsets;

  for (auto &mapEntry : categoryMap) {
    for (InfoInputCategory &catInfo : mapEntry.second) {
      if (!catInfo.wasMerged) {
        continue;
      }
      eraseISec(catInfo.catListIsec);
      catListToErasedOffsets[catInfo.catListIsec].insert(
          catInfo.offCatListIsec);
    }
  }

  // If there were categories that we did not erase, we need to generate a new
  // __objc_catList that contains only the un-merged categories, and get rid of
  // the references to the ones we merged.
  generateCatListForNonErasedCategories(catListToErasedOffsets);

  // Erase the old method lists & names of the categories that were merged
  for (auto &mapEntry : categoryMap) {
    for (InfoInputCategory &catInfo : mapEntry.second) {
      if (!catInfo.wasMerged)
        continue;

      eraseISec(catInfo.catBodyIsec);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec, catLayout.nameOffset,
                                  /*stringOnly=*/true);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec,
                                  catLayout.instanceMethodsOffset);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec,
                                  catLayout.classMethodsOffset);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec,
                                  catLayout.protocolsOffset);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec,
                                  catLayout.classPropsOffset);
      tryEraseDefinedAtIsecOffset(catInfo.catBodyIsec,
                                  catLayout.instancePropsOffset);
    }
  }
}

void ObjcCategoryMerger::doMerge() {
  collectAndValidateCategoriesData();

  for (auto &entry : categoryMap) {
    if (entry.second.size() > 1) {
      // Merge all categories into a new, single category
      mergeCategoriesIntoSingleCategory(entry.second);
      for (auto &catInfo : entry.second) {
        catInfo.wasMerged = true;
      }
    }
  }

  // Erase all categories that were merged
  eraseMergedCategories();
}

void ObjcCategoryMerger::doCleanup() {
  generatedNames.clear();
  generatedSectionData.clear();
}

} // namespace

void objc::mergeCategories() {
  TimeTraceScope timeScope("ObjcCategoryMerger");

  ObjcCategoryMerger merger(inputSections);
  merger.doMerge();
}

void objc::doCleanup() { ObjcCategoryMerger::doCleanup(); }
