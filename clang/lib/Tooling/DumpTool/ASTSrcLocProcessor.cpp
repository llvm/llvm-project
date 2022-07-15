//===- ASTSrcLocProcessor.cpp --------------------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTSrcLocProcessor.h"

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace clang::tooling;
using namespace llvm;
using namespace clang::ast_matchers;

ASTSrcLocProcessor::ASTSrcLocProcessor(StringRef JsonPath)
    : JsonPath(JsonPath) {

  MatchFinder::MatchFinderOptions FinderOptions;

  Finder = std::make_unique<MatchFinder>(std::move(FinderOptions));
  Finder->addMatcher(
      cxxRecordDecl(
          isDefinition(),
          isSameOrDerivedFrom(
              namedDecl(
                  hasAnyName(
                      "clang::Stmt", "clang::Decl", "clang::CXXCtorInitializer",
                      "clang::NestedNameSpecifierLoc",
                      "clang::TemplateArgumentLoc", "clang::CXXBaseSpecifier",
                      "clang::DeclarationNameInfo", "clang::TypeLoc"))
                  .bind("nodeClade")),
          optionally(isDerivedFrom(cxxRecordDecl().bind("derivedFrom"))))
          .bind("className"),
      this);
  Finder->addMatcher(
          cxxRecordDecl(isDefinition(), hasAnyName("clang::PointerLikeTypeLoc",
                                                   "clang::TypeofLikeTypeLoc"))
              .bind("templateName"),
      this);
}

std::unique_ptr<clang::ASTConsumer>
ASTSrcLocProcessor::createASTConsumer(clang::CompilerInstance &Compiler,
                                      StringRef File) {
  return Finder->newASTConsumer();
}

llvm::json::Object toJSON(llvm::StringMap<std::vector<StringRef>> const &Obj) {
  using llvm::json::toJSON;

  llvm::json::Object JsonObj;
  for (const auto &Item : Obj) {
    JsonObj[Item.first()] = Item.second;
  }
  return JsonObj;
}

llvm::json::Object toJSON(llvm::StringMap<std::string> const &Obj) {
  using llvm::json::toJSON;

  llvm::json::Object JsonObj;
  for (const auto &Item : Obj) {
    JsonObj[Item.first()] = Item.second;
  }
  return JsonObj;
}

llvm::json::Object toJSON(ClassData const &Obj) {
  llvm::json::Object JsonObj;

  if (!Obj.ASTClassLocations.empty())
    JsonObj["sourceLocations"] = Obj.ASTClassLocations;
  if (!Obj.ASTClassRanges.empty())
    JsonObj["sourceRanges"] = Obj.ASTClassRanges;
  if (!Obj.TemplateParms.empty())
    JsonObj["templateParms"] = Obj.TemplateParms;
  if (!Obj.TypeSourceInfos.empty())
    JsonObj["typeSourceInfos"] = Obj.TypeSourceInfos;
  if (!Obj.TypeLocs.empty())
    JsonObj["typeLocs"] = Obj.TypeLocs;
  if (!Obj.NestedNameLocs.empty())
    JsonObj["nestedNameLocs"] = Obj.NestedNameLocs;
  if (!Obj.DeclNameInfos.empty())
    JsonObj["declNameInfos"] = Obj.DeclNameInfos;
  return JsonObj;
}

llvm::json::Object toJSON(llvm::StringMap<ClassData> const &Obj) {
  using llvm::json::toJSON;

  llvm::json::Object JsonObj;
  for (const auto &Item : Obj)
    JsonObj[Item.first()] = ::toJSON(Item.second);
  return JsonObj;
}

void WriteJSON(StringRef JsonPath, llvm::json::Object &&ClassInheritance,
               llvm::json::Object &&ClassesInClade,
               llvm::json::Object &&ClassEntries) {
  llvm::json::Object JsonObj;

  using llvm::json::toJSON;

  JsonObj["classInheritance"] = std::move(ClassInheritance);
  JsonObj["classesInClade"] = std::move(ClassesInClade);
  JsonObj["classEntries"] = std::move(ClassEntries);

  llvm::json::Value JsonVal(std::move(JsonObj));

  bool WriteChange = false;
  std::string OutString;
  if (auto ExistingOrErr = MemoryBuffer::getFile(JsonPath, /*IsText=*/true)) {
    raw_string_ostream Out(OutString);
    Out << formatv("{0:2}", JsonVal);
    if (ExistingOrErr.get()->getBuffer() == Out.str())
      return;
    WriteChange = true;
  }

  std::error_code EC;
  llvm::raw_fd_ostream JsonOut(JsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  if (WriteChange)
    JsonOut << OutString;
  else
    JsonOut << formatv("{0:2}", JsonVal);
}

void ASTSrcLocProcessor::generate() {
  WriteJSON(JsonPath, ::toJSON(ClassInheritance), ::toJSON(ClassesInClade),
            ::toJSON(ClassEntries));
}

void ASTSrcLocProcessor::generateEmpty() { WriteJSON(JsonPath, {}, {}, {}); }

std::vector<std::string>
CaptureMethods(std::string TypeString, const clang::CXXRecordDecl *ASTClass,
               const MatchFinder::MatchResult &Result) {

  auto publicAccessor = [](auto... InnerMatcher) {
    return cxxMethodDecl(isPublic(), parameterCountIs(0), isConst(),
                         InnerMatcher...);
  };

  auto BoundNodesVec = match(
      findAll(
          publicAccessor(
              ofClass(cxxRecordDecl(
                  equalsNode(ASTClass),
                  optionally(isDerivedFrom(
                      cxxRecordDecl(hasAnyName("clang::Stmt", "clang::Decl"))
                          .bind("stmtOrDeclBase"))),
                  optionally(isDerivedFrom(
                      cxxRecordDecl(hasName("clang::Expr")).bind("exprBase"))),
                  optionally(
                      isDerivedFrom(cxxRecordDecl(hasName("clang::TypeLoc"))
                                        .bind("typeLocBase"))))),
              returns(hasCanonicalType(asString(TypeString))))
              .bind("classMethod")),
      *ASTClass, *Result.Context);

  std::vector<std::string> Methods;
  for (const auto &BN : BoundNodesVec) {
    if (const auto *Node = BN.getNodeAs<clang::NamedDecl>("classMethod")) {
      const auto *StmtOrDeclBase =
          BN.getNodeAs<clang::CXXRecordDecl>("stmtOrDeclBase");
      const auto *TypeLocBase =
          BN.getNodeAs<clang::CXXRecordDecl>("typeLocBase");
      const auto *ExprBase = BN.getNodeAs<clang::CXXRecordDecl>("exprBase");
      // The clang AST has several methods on base classes which are overriden
      // pseudo-virtually by derived classes.
      // We record only the pseudo-virtual methods on the base classes to
      // avoid duplication.
      if (StmtOrDeclBase &&
          (Node->getName() == "getBeginLoc" || Node->getName() == "getEndLoc" ||
           Node->getName() == "getSourceRange"))
        continue;
      if (ExprBase && Node->getName() == "getExprLoc")
        continue;
      if (TypeLocBase && Node->getName() == "getLocalSourceRange")
        continue;
      if ((ASTClass->getName() == "PointerLikeTypeLoc" ||
           ASTClass->getName() == "TypeofLikeTypeLoc") &&
          Node->getName() == "getLocalSourceRange")
        continue;
      Methods.push_back(Node->getName().str());
    }
  }
  return Methods;
}

void ASTSrcLocProcessor::run(const MatchFinder::MatchResult &Result) {

  const auto *ASTClass =
      Result.Nodes.getNodeAs<clang::CXXRecordDecl>("className");

  StringRef CladeName;
  if (ASTClass) {
    if (const auto *NodeClade =
            Result.Nodes.getNodeAs<clang::CXXRecordDecl>("nodeClade"))
      CladeName = NodeClade->getName();
  } else {
    ASTClass = Result.Nodes.getNodeAs<clang::CXXRecordDecl>("templateName");
    CladeName = "TypeLoc";
  }

  StringRef ClassName = ASTClass->getName();

  ClassData CD;

  CD.ASTClassLocations =
      CaptureMethods("class clang::SourceLocation", ASTClass, Result);
  CD.ASTClassRanges =
      CaptureMethods("class clang::SourceRange", ASTClass, Result);
  CD.TypeSourceInfos =
      CaptureMethods("class clang::TypeSourceInfo *", ASTClass, Result);
  CD.TypeLocs = CaptureMethods("class clang::TypeLoc", ASTClass, Result);
  CD.NestedNameLocs =
      CaptureMethods("class clang::NestedNameSpecifierLoc", ASTClass, Result);
  CD.DeclNameInfos =
      CaptureMethods("struct clang::DeclarationNameInfo", ASTClass, Result);
  auto DI = CaptureMethods("const struct clang::DeclarationNameInfo &",
                           ASTClass, Result);
  CD.DeclNameInfos.insert(CD.DeclNameInfos.end(), DI.begin(), DI.end());

  if (const auto *DerivedFrom =
          Result.Nodes.getNodeAs<clang::CXXRecordDecl>("derivedFrom")) {

    if (const auto *Templ =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                DerivedFrom)) {

      const auto &TArgs = Templ->getTemplateArgs();

      SmallString<256> TArgsString;
      llvm::raw_svector_ostream OS(TArgsString);
      OS << DerivedFrom->getName() << '<';

      clang::PrintingPolicy PPol(Result.Context->getLangOpts());
      PPol.TerseOutput = true;

      for (unsigned I = 0; I < TArgs.size(); ++I) {
        if (I > 0)
          OS << ", ";
        TArgs.get(I).getAsType().print(OS, PPol);
      }
      OS << '>';

      ClassInheritance[ClassName] = TArgsString.str().str();
    } else {
      ClassInheritance[ClassName] = DerivedFrom->getName().str();
    }
  }

  if (const auto *Templ = ASTClass->getDescribedClassTemplate()) {
    if (auto *TParams = Templ->getTemplateParameters()) {
      for (const auto &TParam : *TParams) {
        CD.TemplateParms.push_back(TParam->getName().str());
      }
    }
  }

  ClassEntries[ClassName] = CD;
  ClassesInClade[CladeName].push_back(ClassName);
}
