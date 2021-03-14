//===- ASTSrcLocProcessor.cpp --------------------------------*- C++ -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTSrcLocProcessor.h"

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/JSON.h"

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
              // TODO: Extend this with other clades
              namedDecl(hasName("clang::Stmt")).bind("nodeClade")),
          optionally(isDerivedFrom(cxxRecordDecl().bind("derivedFrom"))))
          .bind("className"),
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

llvm::json::Object toJSON(llvm::StringMap<StringRef> const &Obj) {
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
  return JsonObj;
}

llvm::json::Object toJSON(llvm::StringMap<ClassData> const &Obj) {
  using llvm::json::toJSON;

  llvm::json::Object JsonObj;
  for (const auto &Item : Obj) {
    if (!Item.second.isEmpty())
      JsonObj[Item.first()] = ::toJSON(Item.second);
  }
  return JsonObj;
}

void WriteJSON(std::string JsonPath,
               llvm::StringMap<StringRef> const &ClassInheritance,
               llvm::StringMap<std::vector<StringRef>> const &ClassesInClade,
               llvm::StringMap<ClassData> const &ClassEntries) {
  llvm::json::Object JsonObj;

  using llvm::json::toJSON;

  JsonObj["classInheritance"] = ::toJSON(ClassInheritance);
  JsonObj["classesInClade"] = ::toJSON(ClassesInClade);
  JsonObj["classEntries"] = ::toJSON(ClassEntries);

  std::error_code EC;
  llvm::raw_fd_ostream JsonOut(JsonPath, EC, llvm::sys::fs::F_Text);
  if (EC)
    return;

  llvm::json::Value JsonVal(std::move(JsonObj));
  JsonOut << formatv("{0:2}", JsonVal);
}

void ASTSrcLocProcessor::generate() {
  WriteJSON(JsonPath, ClassInheritance, ClassesInClade, ClassEntries);
}

std::vector<std::string>
CaptureMethods(std::string TypeString, const clang::CXXRecordDecl *ASTClass,
               const MatchFinder::MatchResult &Result) {

  auto publicAccessor = [](auto... InnerMatcher) {
    return cxxMethodDecl(isPublic(), parameterCountIs(0), isConst(),
                         InnerMatcher...);
  };

  auto BoundNodesVec =
      match(findAll(publicAccessor(ofClass(equalsNode(ASTClass)),
                                   returns(asString(TypeString)))
                        .bind("classMethod")),
            *ASTClass, *Result.Context);

  std::vector<std::string> Methods;
  for (const auto &BN : BoundNodesVec) {
    if (const auto *Node = BN.getNodeAs<clang::NamedDecl>("classMethod")) {
      // Only record the getBeginLoc etc on Stmt etc, because it will call
      // more-derived implementations pseudo-virtually.
      if ((ASTClass->getName() != "Stmt" && ASTClass->getName() != "Decl") &&
          (Node->getName() == "getBeginLoc" || Node->getName() == "getEndLoc" ||
           Node->getName() == "getSourceRange")) {
        continue;
      }
      // Only record the getExprLoc on Expr, because it will call
      // more-derived implementations pseudo-virtually.
      if (ASTClass->getName() != "Expr" && Node->getName() == "getExprLoc") {
        continue;
      }
      Methods.push_back(Node->getName().str());
    }
  }
  return Methods;
}

void ASTSrcLocProcessor::run(const MatchFinder::MatchResult &Result) {

  if (const auto *ASTClass =
          Result.Nodes.getNodeAs<clang::CXXRecordDecl>("className")) {

    StringRef ClassName = ASTClass->getName();

    ClassData CD;

    const auto *NodeClade =
        Result.Nodes.getNodeAs<clang::CXXRecordDecl>("nodeClade");
    StringRef CladeName = NodeClade->getName();

    if (const auto *DerivedFrom =
            Result.Nodes.getNodeAs<clang::CXXRecordDecl>("derivedFrom"))
      ClassInheritance[ClassName] = DerivedFrom->getName();

    CD.ASTClassLocations =
        CaptureMethods("class clang::SourceLocation", ASTClass, Result);
    CD.ASTClassRanges =
        CaptureMethods("class clang::SourceRange", ASTClass, Result);

    if (!CD.isEmpty()) {
      ClassEntries[ClassName] = CD;
      ClassesInClade[CladeName].push_back(ClassName);
    }
  }
}
