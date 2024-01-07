#include "clang-symbolgraph-merger/SymbolGraphMerger.h"
#include "clang/AST/DeclObjC.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace llvm;
using namespace llvm::json;
using namespace clang;
using namespace clang::extractapi;
using namespace sgmerger;

namespace {
ObjCInstanceVariableRecord::AccessControl
getAccessFromString(const StringRef AccessLevel) {
  if (AccessLevel.equals("Private"))
    return ObjCInstanceVariableRecord::AccessControl::Private;
  if (AccessLevel.equals("Protected"))
    return ObjCInstanceVariableRecord::AccessControl::Protected;
  if (AccessLevel.equals("Public"))
    return ObjCInstanceVariableRecord::AccessControl::Public;
  if (AccessLevel.equals("Package"))
    return ObjCInstanceVariableRecord::AccessControl::Package;
  return ObjCInstanceVariableRecord::AccessControl::None;
}

Language getLanguageFromString(const StringRef LangName) {
  if (LangName.equals("c"))
    return Language::C;
  if (LangName.equals("objective-c"))
    return Language::ObjC;
  if (LangName.equals("C++"))
    return Language::CXX;

  return Language::Unknown;
}

template <typename Lambda>
bool addWithContainerRecord(APIRecord::RecordKind Kind, APIRecord *TargetRecord,
                            Lambda Inserter) {
  switch (Kind) {
  case APIRecord::RK_ObjCInterface: {
    if (ObjCInterfaceRecord *Container =
            dyn_cast_or_null<ObjCInterfaceRecord>(TargetRecord))
      Inserter(Container);
  } break;
  case APIRecord::RK_ObjCProtocol: {
    if (ObjCProtocolRecord *Container =
            dyn_cast_or_null<ObjCProtocolRecord>(TargetRecord))
      Inserter(Container);
  } break;
  case APIRecord::RK_ObjCCategory: {
    if (ObjCCategoryRecord *Container =
            dyn_cast_or_null<ObjCCategoryRecord>(TargetRecord))
      Inserter(Container);
  } break;
  default:
    return false;
  }
  return true;
}
} // namespace

bool SymbolGraphMerger::merge() {
  for (const auto &SG : SymbolGraphs)
    traverseSymbolGraph(SG);
  return true;
}

bool SymbolGraphMerger::visitMetadata(const Object &Metadata) {
  // TODO: check if all the symbol graphs are generated form same
  // generator or not. Info from metadata is currently not needed to
  // construct the APISet,
  return true;
}

bool SymbolGraphMerger::visitModule(const Object &Module) {
  if (!API) {
    // If the product name is not provided via command line then extract product
    // name from SymbolGraph
    if (ProductName.empty())
      if (auto NameStr = Module.getString("name"))
        ProductName = NameStr->str();

    // extract target triple info
    if (const auto *Platform = Module.getObject("platform")) {
      auto Arch = Platform->getString("architecture");
      const auto *OSObj = Platform->getObject("operatingSystem");
      auto Vendor = Platform->getString("vendor");
      if (!(Arch && Vendor && OSObj))
        return false;
      if (auto OSStr = OSObj->getString("name")) {
        Target = llvm::Triple(*Arch, *Vendor, *OSStr);
        return true;
      }
    }
    return false;
  }
  return true;
}

bool SymbolGraphMerger::visitRelationship(const Object &Relationship) {
  std::string SourceUSR(Relationship.getString("source").value_or(""));
  std::string TargetUSR(Relationship.getString("target").value_or(""));

  auto *SourceSymbol = PendingSymbols.lookup(SourceUSR);
  auto *TargetSymbol = VisitedSymbols.lookup(TargetUSR);
  auto *TargetRecord = API->findRecordForUSR(TargetUSR);

  switch (SourceSymbol->Kind) {
  case APIRecord::RK_StructField: {
    if (StructRecord *ParentStruct =
            dyn_cast_or_null<StructRecord>(TargetRecord))
      API->addStructField(ParentStruct, SourceSymbol->Name, SourceSymbol->USR,
                          SourceSymbol->Location, SourceSymbol->Availabilities,
                          SourceSymbol->Comments, SourceSymbol->DeclFragments,
                          SourceSymbol->SubHeadings,
                          false /*IsFromSystemHeader*/);
  } break;
  case APIRecord::RK_ObjCIvar: {
    auto AddRecord = [&](ObjCContainerRecord *Container) {
      ObjCInstanceVariableRecord::AccessControl Access =
          getAccessFromString(SourceSymbol->AccessLevel);

      API->addObjCInstanceVariable(
          Container, SourceSymbol->Name, SourceSymbol->USR,
          SourceSymbol->Location, SourceSymbol->Availabilities,
          SourceSymbol->Comments, SourceSymbol->DeclFragments,
          SourceSymbol->SubHeadings, Access, false /*IsFromSystemHeader*/);
    };
    if (!addWithContainerRecord(TargetSymbol->Kind, TargetRecord, AddRecord))
      return false;
  } break;
  case APIRecord::RK_ObjCInstanceMethod: {
    auto AddRecord = [&](ObjCContainerRecord *Container) {
      API->addObjCMethod(Container, SourceSymbol->Name, SourceSymbol->USR,
                         SourceSymbol->Location, SourceSymbol->Availabilities,
                         SourceSymbol->Comments, SourceSymbol->DeclFragments,
                         SourceSymbol->SubHeadings, SourceSymbol->FunctionSign,
                         true
                         /*IsInstanceMethod*/,
                         false /*IsFromSystemHeader*/);
    };
    if (!addWithContainerRecord(TargetSymbol->Kind, TargetRecord, AddRecord))
      return false;
  } break;
  case APIRecord::RK_EnumConstant: {
    if (EnumRecord *Enum = dyn_cast_or_null<EnumRecord>(TargetRecord))
      API->addEnumConstant(Enum, SourceSymbol->Name, SourceSymbol->USR,
                           SourceSymbol->Location, SourceSymbol->Availabilities,
                           SourceSymbol->Comments, SourceSymbol->DeclFragments,
                           SourceSymbol->SubHeadings,
                           false /*IsFromSystemHeader*/);
  } break;
  case APIRecord::RK_ObjCClassMethod: {
    auto AddRecord = [&](ObjCContainerRecord *Container) {
      API->addObjCMethod(Container, SourceSymbol->Name, SourceSymbol->USR,
                         SourceSymbol->Location, SourceSymbol->Availabilities,
                         SourceSymbol->Comments, SourceSymbol->DeclFragments,
                         SourceSymbol->SubHeadings, SourceSymbol->FunctionSign,
                         false
                         /*IsInstanceMethod*/,
                         false /*IsFromSystemHeader*/);
    };
    if (!addWithContainerRecord(TargetSymbol->Kind, TargetRecord, AddRecord))
      return false;
  } break;
  case APIRecord::RK_ObjCInstanceProperty: {
    auto AddRecord = [&](ObjCContainerRecord *Container) {
      API->addObjCProperty(Container, SourceSymbol->Name, SourceSymbol->USR,
                           SourceSymbol->Location, SourceSymbol->Availabilities,
                           SourceSymbol->Comments, SourceSymbol->DeclFragments,
                           SourceSymbol->SubHeadings,
                           ObjCPropertyRecord::AttributeKind::NoAttr, "", "",
                           false /*IsOptional*/, true /*IsInstanceProperty*/,
                           false /*IsFromSystemHeader*/);
    };
    if (!addWithContainerRecord(TargetSymbol->Kind, TargetRecord, AddRecord))
      return false;
  } break;
  case APIRecord::RK_ObjCClassProperty: {
    auto AddRecord = [&](ObjCContainerRecord *Container) {
      API->addObjCProperty(Container, SourceSymbol->Name, SourceSymbol->USR,
                           SourceSymbol->Location, SourceSymbol->Availabilities,
                           SourceSymbol->Comments, SourceSymbol->DeclFragments,
                           SourceSymbol->SubHeadings,
                           ObjCPropertyRecord::AttributeKind::NoAttr, "", "",
                           false /*IsOptional*/, false /*IsInstanceProperty*/,
                           false /*IsFromSystemHeader*/);
    };
    if (!addWithContainerRecord(TargetSymbol->Kind, TargetRecord, AddRecord))
      return false;
  } break;
  case APIRecord::RK_ObjCInterface: {
    if (TargetRecord) {
      SymbolReference SuperClass(TargetRecord);

      API->addObjCInterface(
          SourceSymbol->Name, SourceSymbol->USR, SourceSymbol->Location,
          SourceSymbol->Availabilities, LinkageInfo(), SourceSymbol->Comments,
          SourceSymbol->DeclFragments, SourceSymbol->SubHeadings, SuperClass,
          false /*IsFromSystemHeader*/);
    }
  } break;
  case APIRecord::RK_ObjCCategory: {
    if (TargetRecord) {
      SymbolReference Interface(TargetRecord);

      API->addObjCCategory(
          SourceSymbol->Name, SourceSymbol->USR, SourceSymbol->Location,
          SourceSymbol->Availabilities, SourceSymbol->Comments,
          SourceSymbol->DeclFragments, SourceSymbol->SubHeadings, Interface,
          false /*IsFromSystemHeader*/, true /*IsFromExternalModule*/);
    }
  } break;
  default:
    return false;
  }
  return true;
}

bool SymbolGraphMerger::visitSymbol(const SymbolGraph::Symbol &Symbol) {
  // If the APISet is not yet created, then create it ( it's generally
  // the Language information that is pending uptill this point )
  if (!API) {
    if (const auto *Id = Symbol.SymbolObj.getObject("identifier"))
      if (const auto LangName = Id->getString("interfaceLanguage"))
        Lang = getLanguageFromString(LangName.value_or(""));
    API = std::make_unique<extractapi::APISet>(Target, Lang, ProductName);
  }

  switch (Symbol.Kind) {
  // TODO: Handle unknown symbols properly
  case APIRecord::RK_Unknown:
    break;
  case APIRecord::RK_GlobalVariable: {
    API->addGlobalVar(Symbol.Name, Symbol.USR, Symbol.Location,
                      Symbol.Availabilities, LinkageInfo(), Symbol.Comments,
                      Symbol.DeclFragments, Symbol.SubHeadings,
                      false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  case APIRecord::RK_GlobalFunction: {
    API->addGlobalFunction(
        Symbol.Name, Symbol.USR, Symbol.Location, Symbol.Availabilities,
        LinkageInfo(), Symbol.Comments, Symbol.DeclFragments,
        Symbol.SubHeadings, Symbol.FunctionSign, false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;

  case APIRecord::RK_Enum: {
    API->addEnum(Symbol.Name, Symbol.USR, Symbol.Location,
                 Symbol.Availabilities, Symbol.Comments, Symbol.DeclFragments,
                 Symbol.SubHeadings, false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  case APIRecord::RK_Struct: {
    API->addStruct(Symbol.Name, Symbol.USR, Symbol.Location,
                   Symbol.Availabilities, Symbol.Comments, Symbol.DeclFragments,
                   Symbol.SubHeadings, false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  case APIRecord::RK_ObjCProtocol: {
    API->addObjCProtocol(Symbol.Name, Symbol.USR, Symbol.Location,
                         Symbol.Availabilities, Symbol.Comments,
                         Symbol.DeclFragments, Symbol.SubHeadings,
                         false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  case APIRecord::RK_MacroDefinition: {
    API->addMacroDefinition(Symbol.Name, Symbol.USR, Symbol.Location,
                            Symbol.DeclFragments, Symbol.SubHeadings,
                            false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  case APIRecord::RK_Typedef: {
    API->addTypedef(Symbol.Name, Symbol.USR, Symbol.Location,
                    Symbol.Availabilities, Symbol.Comments,
                    Symbol.DeclFragments, Symbol.SubHeadings,
                    Symbol.UnderLyingType, false /*IsFromSystemHeader*/);
    VisitedSymbols[Symbol.USR] = &Symbol;
  } break;
  default:
    // Try again when visiting Relationships
    PendingSymbols[Symbol.USR] = &Symbol;
  }
  return true;
}
