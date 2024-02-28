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
  DO(uint32_t, size)                                                           \
  DO(uint32_t, padding)

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

// \p r must point to an offset within a cstring section or ConcatInputSection
static StringRef getReferentString(const Reloc &r) {
  if (auto *isec = r.referent.dyn_cast<InputSection *>())
    return cast<CStringInputSection>(isec)->getStringRefAtOffset(r.addend);
  auto *sym = cast<Defined>(r.referent.get<Symbol *>());
  uint32_t dataOff = sym->value + r.addend;
  if (auto *cisec = dyn_cast<ConcatInputSection>(sym->isec)) {
    uint32_t buffSize = cisec->data.size();
    const char *pszBuff = reinterpret_cast<const char *>(cisec->data.data());
    assert(dataOff < buffSize);
    uint32_t sLen = strnlen(pszBuff + dataOff, buffSize - dataOff);
    llvm::StringRef strRef(pszBuff + dataOff, sLen);
    assert(strRef.size() > 0 && "getReferentString returning empty string");
    return strRef;
  }
  return cast<CStringInputSection>(sym->isec)->getStringRefAtOffset(dataOff);
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
    const Section *inputSection;
    Reloc relocTemplate;
    T *outputSection;
  };

  struct InfoCategoryWriter {
    InfroWriteSection<ConcatOutputSection> catListInfo;
    InfroWriteSection<CStringSection> catNameInfo;
    InfroWriteSection<ConcatOutputSection> catBodyInfo;
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

    PointerListInfo instanceMethods = "__OBJC_$_CATEGORY_INSTANCE_METHODS_";
    PointerListInfo classMethods = "__OBJC_$_CATEGORY_CLASS_METHODS_";
    PointerListInfo protocols = "__OBJC_CATEGORY_PROTOCOLS_$_";
    PointerListInfo instanceProps = "__OBJC_$_PROP_LIST_";
    PointerListInfo classProps = "__OBJC_$_CLASS_PROP_LIST_";
  };

public:
  ObjcCategoryMerger(std::vector<ConcatInputSection *> &_allInputSections);
  bool doMerge();

private:
  // This returns bool and always false for easy 'return false;' statements
  bool registerError(const char *msg);

  bool collectAndValidateCategoriesData();
  bool
  mergeCategoriesIntoSingleCategory(std::vector<InfoInputCategory> &categories);
  bool eraseMergedCategories();

  bool generateCatListForNonErasedCategories(
      std::map<ConcatInputSection *, std::set<uint64_t>>
          catListToErasedOffsets);
  template <typename T>
  bool collectSectionWriteInfoFromIsec(InputSection *isec,
                                       InfroWriteSection<T> &catWriteInfo);
  bool collectCategoryWriterInfoFromCategory(InfoInputCategory &catInfo);
  bool parseCatInfoToExtInfo(InfoInputCategory &catInfo,
                             ClassExtensionInfo &extInfo);

  bool tryParseProtocolListInfo(ConcatInputSection *isec,
                                uint32_t symbolsPerStruct,
                                PointerListInfo &ptrList);

  bool parsePointerListInfo(ConcatInputSection *isec, uint32_t secOffset,
                            uint32_t symbolsPerStruct,
                            PointerListInfo &ptrList);

  bool emitAndLinkPointerList(Defined *parentSym, uint32_t linkAtOffset,
                              ClassExtensionInfo &extInfo,
                              PointerListInfo &ptrList);

  bool emitAndLinkProtocolList(Defined *parentSym, uint32_t linkAtOffset,
                               ClassExtensionInfo &extInfo,
                               PointerListInfo &ptrList);

  bool emitCategory(ClassExtensionInfo &extInfo, Defined *&catBodySym);
  bool emitCatListEntrySec(std::string &forCateogryName,
                           std::string &forBaseClassName, ObjFile *objFile,
                           Defined *&catListSym);
  bool emitCategoryBody(std::string &name, Defined *nameSym,
                        Symbol *baseClassSym, std::string &baseClassName,
                        ObjFile *objFile, Defined *&catBodySym);
  bool emitCategoryName(std::string &name, ObjFile *objFile,
                        Defined *&catNameSym);
  bool createSymbolReference(Defined *refFrom, Symbol *refTo, uint32_t offset,
                             Reloc &relocTemplate);
  bool tryGetSymbolAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
                                Symbol *&sym);
  bool tryGetDefinedAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
                                 Defined *&defined);
  bool tryEraseDefinedAtIsecOffset(ConcatInputSection *isec, uint32_t offset,
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

bool ObjcCategoryMerger::registerError(const char *msg) {
  std::string err = "ObjC category merging error[-merge-objc-categories]: ";
  err += msg;
  error(err);
  return false; // Always return false for easy 'return registerError()' syntax.
}

// This is a template so that it can be used both for CStringSection and
// ConcatOutputSection
template <typename T>
bool ObjcCategoryMerger::collectSectionWriteInfoFromIsec(
    InputSection *isec, InfroWriteSection<T> &catWriteInfo) {
  if (catWriteInfo.valid)
    return true;

  catWriteInfo.inputSection = &isec->section;
  catWriteInfo.align = isec->align;
  catWriteInfo.outputSection = dyn_cast_or_null<T>(isec->parent);

  if (isec->relocs.size())
    catWriteInfo.relocTemplate = isec->relocs[0];

  if (!catWriteInfo.outputSection) {
    std::string message =
        "Unexpected output section type for" + isec->getName().str();
    return registerError(message.c_str());
  }

  catWriteInfo.valid = true;

  return true;
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
bool ObjcCategoryMerger::tryEraseDefinedAtIsecOffset(ConcatInputSection *isec,
                                                     uint32_t offset,
                                                     bool stringOnly) {
  const Reloc *reloc = isec->getRelocAt(offset);

  if (!reloc)
    return false;

  Defined *sym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());

  if (!sym)
    return false;

  auto *cisec = dyn_cast_or_null<ConcatInputSection>(sym->isec);
  if (!stringOnly && cisec) {
    cisec->linkerOptimizeReason = LinkerOptReason::CategoryMerging;
    return true;
  }

  if (auto *cisec = dyn_cast_or_null<CStringInputSection>(sym->isec)) {
    uint32_t totalOffset = sym->value + reloc->addend;
    StringPiece &piece = cisec->getStringPiece(totalOffset);
    piece.linkerOptimizeReason = LinkerOptReason::CategoryMerging;
    return true;
  }

  return false;
}

bool ObjcCategoryMerger::collectCategoryWriterInfoFromCategory(
    InfoInputCategory &catInfo) {

  if (!collectSectionWriteInfoFromIsec<ConcatOutputSection>(
          catInfo.catListIsec, infoCategoryWriter.catListInfo))
    return false;
  if (!collectSectionWriteInfoFromIsec<ConcatOutputSection>(
          catInfo.catBodyIsec, infoCategoryWriter.catBodyInfo))
    return false;

  if (!infoCategoryWriter.catNameInfo.valid) {
    const Reloc *catNameReloc =
        catInfo.catBodyIsec->getRelocAt(catLayout.nameOffset);

    if (!catNameReloc)
      return registerError("Category does not have a reloc at nameOffset");

    lld::macho::Defined *catDefSym =
        dyn_cast_or_null<Defined>(catNameReloc->referent.dyn_cast<Symbol *>());
    if (!catDefSym)
      return registerError(
          "Reloc of category name is not a valid Defined symbol");

    if (!collectSectionWriteInfoFromIsec<CStringSection>(
            catDefSym->isec, infoCategoryWriter.catNameInfo))
      return false;
  }

  // Collect writer info from all the category lists (we're assuming they all
  // would provide the same info)
  if (!infoCategoryWriter.catPtrListInfo.valid) {
    for (uint32_t off = catLayout.instanceMethodsOffset;
         off <= catLayout.classPropsOffset; off += target->wordSize) {
      Defined *ptrList;
      if (tryGetDefinedAtIsecOffset(catInfo.catBodyIsec, off, ptrList)) {
        if (!collectSectionWriteInfoFromIsec<ConcatOutputSection>(
                ptrList->isec, infoCategoryWriter.catPtrListInfo))
          return false;
        break;
      }
    }
  }

  return true;
}

// Parse a protocol list that might be linked to at a ConcatInputSection given
// offset. The format of the protocol list is different than other lists (prop
// lists, method lists) so we need to parse it differently
bool ObjcCategoryMerger::tryParseProtocolListInfo(ConcatInputSection *isec,
                                                  uint32_t secOffset,
                                                  PointerListInfo &ptrList) {
  if (!isec || (secOffset + target->wordSize > isec->data.size()))
    return registerError(
        "Tried to read pointer list beyond protocol section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  if (!reloc)
    return true; // List is null, return true because no m_error

  auto *ptrListSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
  if (!ptrListSym)
    return registerError("Protocol list reloc does not have a valid Defined");

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
  if (expectedListSize != ptrListSym->isec->data.size())
    return registerError("Protocol list does not match expected size");

  uint32_t off = protocolListHeaderLayout.totalSize;
  for (uint32_t inx = 0; inx < protocolCount; inx++) {
    const Reloc *reloc = ptrListSym->isec->getRelocAt(off);
    if (!reloc)
      return registerError("No reloc found at protocol list offset");

    auto *listSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
    if (!listSym)
      return registerError("Protocol list reloc does not have a valid Defined");

    ptrList.allPtrs.push_back(listSym);
    off += target->wordSize;
  }

  return true;
}

// Parse a pointer list that might be linked to at a ConcatInputSection given
// offset. This can be used for instance methods, class methods, instance props
// and class props since they have the same format.
bool ObjcCategoryMerger::parsePointerListInfo(ConcatInputSection *isec,
                                              uint32_t secOffset,
                                              uint32_t symbolsPerStruct,
                                              PointerListInfo &ptrList) {
  assert(symbolsPerStruct == 2 || symbolsPerStruct == 3);
  if (!isec || (secOffset + target->wordSize > isec->data.size()))
    return registerError("Tried to read pointer list beyond section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  if (!reloc)
    return true; // No reloc found, nothing to parse, so return success

  auto *ptrListSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
  if (!ptrListSym)
    return registerError("Reloc does not have a valid Defined");

  uint32_t thisStructSize = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec->data.data() + listHeaderLayout.structSizeOffset);
  uint32_t thisStructCount = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec->data.data() + listHeaderLayout.structCountOffset);

  assert(!ptrList.structSize || (thisStructSize == ptrList.structSize));

  ptrList.structCount += thisStructCount;
  ptrList.structSize = thisStructSize;

  uint32_t expectedListSize =
      listHeaderLayout.totalSize + (thisStructSize * thisStructCount);

  if (expectedListSize != ptrListSym->isec->data.size())
    return registerError("Pointer list does not match expected size");

  for (uint32_t off = listHeaderLayout.totalSize; off < expectedListSize;
       off += target->wordSize) {
    const Reloc *reloc = ptrListSym->isec->getRelocAt(off);
    if (!reloc)
      return registerError("No reloc found at pointer list offset");

    auto *listSym = dyn_cast_or_null<Defined>(reloc->referent.get<Symbol *>());
    if (!listSym)
      return registerError("Reloc does not have a valid Defined");

    ptrList.allPtrs.push_back(listSym);
  }

  return true;
}

// Here we parse all the information of an input category (catInfo) and
// append-store the parsed info into the strucutre which will contain all the
// information about how a class is extended (extInfo)
bool ObjcCategoryMerger::parseCatInfoToExtInfo(InfoInputCategory &catInfo,
                                               ClassExtensionInfo &extInfo) {
  const Reloc *catNameReloc =
      catInfo.catBodyIsec->getRelocAt(catLayout.nameOffset);

  //// Parse name ///////////////////////////////////////////////////////////
  if (!catNameReloc)
    return registerError("Category does not have a reloc at 'nameOffset'");

  if (!extInfo.mergedContainerName.empty())
    extInfo.mergedContainerName += "|";

  if (!extInfo.objFileForMergeData)
    extInfo.objFileForMergeData =
        dyn_cast_or_null<ObjFile>(catInfo.catBodyIsec->getFile());

  StringRef catName = getReferentString(*catNameReloc);
  extInfo.mergedContainerName += catName.str();

  //// Parse base class /////////////////////////////////////////////////////
  const Reloc *klassReloc =
      catInfo.catBodyIsec->getRelocAt(catLayout.klassOffset);

  if (!klassReloc)
    return registerError("Category does not have a reloc at 'klassOffset'");

  Symbol *classSym = klassReloc->referent.get<Symbol *>();

  if (extInfo.baseClass && extInfo.baseClass != classSym)
    return registerError("Trying to parse category info into container with "
                         "different base class");

  extInfo.baseClass = classSym;

  if (extInfo.baseClassName.empty()) {
    llvm::StringRef classPrefix("_OBJC_CLASS_$_");
    if (!classSym->getName().starts_with(classPrefix))
      return registerError(
          "Base class symbol does not start with '_OBJC_CLASS_$_'");

    extInfo.baseClassName = classSym->getName().substr(classPrefix.size());
  }

  if (!parsePointerListInfo(catInfo.catBodyIsec,
                            catLayout.instanceMethodsOffset,
                            /*symbolsPerStruct=*/3, extInfo.instanceMethods))
    return false;

  if (!parsePointerListInfo(catInfo.catBodyIsec, catLayout.classMethodsOffset,
                            /*symbolsPerStruct=*/3, extInfo.classMethods))
    return false;

  if (!tryParseProtocolListInfo(catInfo.catBodyIsec, catLayout.protocolsOffset,
                                extInfo.protocols))
    return false;

  if (!parsePointerListInfo(catInfo.catBodyIsec, catLayout.instancePropsOffset,
                            /*symbolsPerStruct=*/2, extInfo.instanceProps))
    return false;

  if (!parsePointerListInfo(catInfo.catBodyIsec, catLayout.classPropsOffset,
                            /*symbolsPerStruct=*/2, extInfo.classProps))
    return false;

  return true;
}

// Generate a protocol list (including header) and link it into the parent at
// the specified offset.
bool ObjcCategoryMerger::emitAndLinkProtocolList(Defined *parentSym,
                                                 uint32_t linkAtOffset,
                                                 ClassExtensionInfo &extInfo,
                                                 PointerListInfo &ptrList) {
  if (ptrList.allPtrs.empty())
    return true;

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

  if (!createSymbolReference(parentSym, ptrListSym, linkAtOffset,
                             infoCategoryWriter.catBodyInfo.relocTemplate))
    return false;

  uint32_t offset = protocolListHeaderLayout.totalSize;
  for (Symbol *symbol : ptrList.allPtrs) {
    if (!createSymbolReference(ptrListSym, symbol, offset,
                               infoCategoryWriter.catPtrListInfo.relocTemplate))
      return false;

    offset += target->wordSize;
  }

  return true;
}

// Generate a pointer list (including header) and link it into the parent at the
// specified offset. This is used for instance and class methods and
// proprieties.
bool ObjcCategoryMerger::emitAndLinkPointerList(Defined *parentSym,
                                                uint32_t linkAtOffset,
                                                ClassExtensionInfo &extInfo,
                                                PointerListInfo &ptrList) {
  if (ptrList.allPtrs.empty())
    return true;

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

  if (!createSymbolReference(parentSym, ptrListSym, linkAtOffset,
                             infoCategoryWriter.catBodyInfo.relocTemplate))
    return false;

  uint32_t offset = listHeaderLayout.totalSize;
  for (Symbol *symbol : ptrList.allPtrs) {
    if (!createSymbolReference(ptrListSym, symbol, offset,
                               infoCategoryWriter.catPtrListInfo.relocTemplate))
      return false;

    offset += target->wordSize;
  }

  return true;
}

// This method creates an __objc_catlist ConcatInputSection with a single slot
bool ObjcCategoryMerger::emitCatListEntrySec(std::string &forCateogryName,
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
  return true;
}

// Here we generate the main category body and just the body and link the name
// and base class into it. We don't link any other info like the protocol and
// class/instance methods/props.
bool ObjcCategoryMerger::emitCategoryBody(std::string &name, Defined *nameSym,
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
      "__OBJC_$_CATEGORY_" + baseClassName + "_$_(" + name + ")";
  generatedNames.push_back(StringRef(symName));
  catBodySym = make<Defined>(
      StringRef(generatedNames.back()), /*file=*/objFile, newBodySec,
      /*value=*/0, bodyData.size(), /*isWeakDef=*/false, /*isExternal=*/false,
      /*isPrivateExtern=*/false, /*includeInSymtab=*/true,
      /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
      /*isWeakDefCanBeHidden=*/false);

  catBodySym->used = true;
  objFile->symbols.push_back(catBodySym);

  if (!createSymbolReference(catBodySym, nameSym, catLayout.nameOffset,
                             infoCategoryWriter.catBodyInfo.relocTemplate))
    return false;

  // Create a reloc to the base class (either external or internal)
  if (!createSymbolReference(catBodySym, baseClassSym, catLayout.klassOffset,
                             infoCategoryWriter.catBodyInfo.relocTemplate))
    return false;

  return true;
}

// This writes the new category name (for the merged category) into the binary
// and returns the sybmol for it.
bool ObjcCategoryMerger::emitCategoryName(std::string &name, ObjFile *objFile,
                                          Defined *&catNamdeSym) {
  llvm::ArrayRef<uint8_t> inputNameArrData(
      reinterpret_cast<const uint8_t *>(name.c_str()), name.size() + 1);
  generatedSectionData.push_back(SmallVector<uint8_t>(inputNameArrData));

  llvm::ArrayRef<uint8_t> nameData = generatedSectionData.back();

  CStringInputSection *newStringSec = make<CStringInputSection>(
      *infoCategoryWriter.catNameInfo.inputSection, nameData,
      infoCategoryWriter.catNameInfo.align, true);

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
  return true;
}

// This method fully creates a new category from the given ClassExtensionInfo.
// It creates the category body, name and protocol/method/prop lists an links
// everything together. Then it creates a new __objc_catlist entry and links the
// category into it. Calling this method will fully generate a category which
// will be available in the final binary.
bool ObjcCategoryMerger::emitCategory(ClassExtensionInfo &extInfo,
                                      Defined *&catBodySym) {
  Defined *catNameSym = nullptr;
  if (!emitCategoryName(extInfo.mergedContainerName,
                        extInfo.objFileForMergeData, catNameSym))
    return false;

  if (!emitCategoryBody(extInfo.mergedContainerName, catNameSym,
                        extInfo.baseClass, extInfo.baseClassName,
                        extInfo.objFileForMergeData, catBodySym))
    return false;

  Defined *catListSym = nullptr;
  if (!emitCatListEntrySec(extInfo.mergedContainerName, extInfo.baseClassName,
                           extInfo.objFileForMergeData, catListSym))
    return false;

  const uint32_t offsetFirstCat = 0;
  if (!createSymbolReference(catListSym, catBodySym, offsetFirstCat,
                             infoCategoryWriter.catListInfo.relocTemplate))
    return false;

  if (!emitAndLinkPointerList(catBodySym, catLayout.instanceMethodsOffset,
                              extInfo, extInfo.instanceMethods))
    return false;

  if (!emitAndLinkPointerList(catBodySym, catLayout.classMethodsOffset, extInfo,
                              extInfo.classMethods))
    return false;

  if (!emitAndLinkProtocolList(catBodySym, catLayout.protocolsOffset, extInfo,
                               extInfo.protocols))
    return false;

  if (!emitAndLinkPointerList(catBodySym, catLayout.instancePropsOffset,
                              extInfo, extInfo.instanceProps))
    return false;

  if (!emitAndLinkPointerList(catBodySym, catLayout.classPropsOffset, extInfo,
                              extInfo.classProps))
    return false;

  return true;
}

// This method merges all the categories (sharing a base class) into a single
// category.
bool ObjcCategoryMerger::mergeCategoriesIntoSingleCategory(
    std::vector<InfoInputCategory> &categories) {
  assert(categories.size() > 1 && "Expected at least 2 categories");

  ClassExtensionInfo extInfo;

  for (auto &catInfo : categories)
    if (!parseCatInfoToExtInfo(catInfo, extInfo))
      return false;

  Defined *newCatDef = nullptr;
  if (!emitCategory(extInfo, newCatDef))
    return false;

  return true;
}

bool ObjcCategoryMerger::createSymbolReference(Defined *refFrom, Symbol *refTo,
                                               uint32_t offset,
                                               Reloc &relocTemplate) {
  Reloc r = relocTemplate;
  r.offset = offset;
  r.addend = 0;
  r.referent = refTo;
  refFrom->isec->relocs.push_back(r);

  return true;
}

bool ObjcCategoryMerger::collectAndValidateCategoriesData() {
  for (InputSection *sec : allInputSections) {
    if (sec->getName() != section_names::objcCatList)
      continue;
    ConcatInputSection *catListCisec = dyn_cast<ConcatInputSection>(sec);
    if (!catListCisec)
      return registerError(
          "__objc_catList InputSection is not a ConcatInputSection");

    for (const Reloc &r : catListCisec->relocs) {
      auto *sym = cast<Defined>(r.referent.get<Symbol *>());
      if (!sym || !sym->getName().starts_with("__OBJC_$_CATEGORY_"))
        continue; // Only support ObjC categories (no swift + @objc)

      auto *catBodyIsec =
          dyn_cast<ConcatInputSection>(r.getReferentInputSection());
      if (!catBodyIsec)
        return registerError(
            "Category data section is not an ConcatInputSection");

      if (catBodyIsec->getSize() != catLayout.totalSize) {
        std::string err;
        llvm::raw_string_ostream OS(err);
        OS << "Invalid input category size encountered, category merging only "
              "supports "
           << catLayout.totalSize << " bytes";
        OS.flush();
        return registerError(err.c_str());
      }

      // Check that the category has a reloc at 'klassOffset' (which is
      // a pointer to the class symbol)

      auto *classReloc = catBodyIsec->getRelocAt(catLayout.klassOffset);
      if (!classReloc)
        return registerError("Category does not have a reloc at klassOffset");

      auto *classSym = classReloc->referent.get<Symbol *>();
      InfoInputCategory catInputInfo{catBodyIsec, catListCisec, r.offset};
      categoryMap[classSym].push_back(catInputInfo);

      if (!collectCategoryWriterInfoFromCategory(catInputInfo))
        return false;
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

  return true;
}

// In the input we have multiple __objc_catlist InputSection, each of which may
// contain links to multiple categories. Of these categories, we will merge (and
// erase) only some. There will be some categories that will remain unoutched
// (not erased). For these not erased categories, we generate new __objc_catlist
// entries since the parent __objc_catlist entry will be erased
bool ObjcCategoryMerger::generateCatListForNonErasedCategories(
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
      if (!tryGetDefinedAtIsecOffset(catListIsec, catListIsecOffset,
                                     nonErasedCatBody))
        return registerError("Failed to relocate non-deleted category");

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
      if (!createSymbolReference(catListSlotSym, nonErasedCatBody, 0,
                                 infoCategoryWriter.catListInfo.relocTemplate))
        return registerError(
            "Failed to create symbol reference to non-deleted category");

      catListIsecOffset += target->wordSize;
    }
  }
  return true;
}

// This fully erases the merged categories, including their body, their names,
// their method/protocol/prop lists and the __objc_catlist entries that link to
// them.
bool ObjcCategoryMerger::eraseMergedCategories() {
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
      catInfo.catListIsec->linkerOptimizeReason =
          LinkerOptReason::CategoryMerging;
      catListToErasedOffsets[catInfo.catListIsec].insert(
          catInfo.offCatListIsec);
    }
  }

  // If there were categories that we did not erase, we need to generate a new
  // __objc_catList that contains only the un-merged categories, and get rid of
  // the references to the ones we merged.
  if (!generateCatListForNonErasedCategories(catListToErasedOffsets))
    return false;

  // Erase the old method lists & names of the categories that were merged
  for (auto &mapEntry : categoryMap) {
    for (InfoInputCategory &catInfo : mapEntry.second) {
      if (!catInfo.wasMerged)
        continue;

      catInfo.catBodyIsec->linkerOptimizeReason =
          LinkerOptReason::CategoryMerging;
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

  return true;
}

bool ObjcCategoryMerger::doMerge() {
  if (!collectAndValidateCategoriesData())
    return false;

  for (auto &entry : categoryMap) {
    // Can't merge a single category into the base class just yet.
    if (entry.second.size() <= 1)
      continue;

    // Merge all categories into a new, single category
    if (!mergeCategoriesIntoSingleCategory(entry.second))
      return false;

    for (auto &catInfo : entry.second) {
      catInfo.wasMerged = true;
    }
  }

  // If we reach here, all categories in entry were merged, so mark them
  if (!eraseMergedCategories())
    return false;

  return true;
}

} // namespace

void objc::mergeCategories() {
  TimeTraceScope timeScope("ObjcCategoryMerger");

  ObjcCategoryMerger merger(inputSections);
  merger.doMerge();
}
