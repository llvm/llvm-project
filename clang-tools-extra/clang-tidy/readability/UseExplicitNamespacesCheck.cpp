//===--- UseExplicitNamespacesCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseExplicitNamespacesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

bool isTransparentNamespace(const NamespaceDecl *decl) {
  return decl->isInline() || decl->isAnonymousNamespace();
}

StringRef getIdentifierString(const IdentifierInfo *identifier) {
  return identifier ? identifier->getName() : "(nullptr)";
}

StringRef getMatchContext(StringRef match, const DynTypedNode &node) {
  llvm::SmallString<128> out;
  out.append(match);
  out.append("(");
  out.append(node.getNodeKind().asStringRef());
  out.append(")");
  return out.str();
}

StringRef trueFalseString(bool value) { return value ? "true" : "false"; }

UseExplicitNamespacesCheck::UseExplicitNamespacesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      limitToPattern(Options.get("LimitToPattern", "")),
      onlyExpandUsingNamespace(Options.get("OnlyExpandUsingNamespace", true)),
      diagnosticLevel(Options.get("DiagnosticLevel", 0)) {
  auto limitString = limitToPattern.str();
  size_t startIndex = 0;
  while (startIndex < limitString.size()) {
    size_t found = limitString.find("::", startIndex);
    size_t end = found == std::string::npos ? limitString.size() : found;
    size_t length = end - startIndex;
    if (length) {
      limitToPatternVector.emplace_back(limitString, startIndex, length);
    }
    startIndex = end + 2;
  }
}

void UseExplicitNamespacesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LimitToPattern", limitToPattern);
  Options.store(Opts, "OnlyExpandUsingNamespace", onlyExpandUsingNamespace);
  Options.store(Opts, "DiagnosticLevel", diagnosticLevel);
}

void UseExplicitNamespacesCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      declRefExpr(unless(hasAncestor(substNonTypeTemplateParmExpr())))
          .bind("DeclRefExpr"),
      this);
  Finder->addMatcher(declaratorDecl().bind("DeclaratorDecl"), this);
  Finder->addMatcher(cxxNewExpr().bind("CXXNewExpr"), this);
  Finder->addMatcher(cxxTemporaryObjectExpr().bind("CXXTemporaryObjectExpr"),
                     this);
  Finder->addMatcher(typedefNameDecl().bind("TypedefNameDecl"), this);
}

inline bool
findMatch(size_t &currentTargetIndex,
          const std::vector<const DeclContext *> &targetContextVector,
          const std::string &name) {
  while (currentTargetIndex < targetContextVector.size()) {
    auto currentContext = targetContextVector[currentTargetIndex];
    ++currentTargetIndex;
    if (auto currentNamespace = dyn_cast<NamespaceDecl>(currentContext)) {
      if (currentNamespace->isAnonymousNamespace()) {
        return false;
      }
      if (currentNamespace->getIdentifier()->getName().str() == name) {
        return true;
      }
      if (!currentNamespace->isInline()) {
        return false;
      }
    } else {
      return false;
    }
  }
  return false;
};

bool UseExplicitNamespacesCheck::matchesNamespaceLimits(
    const std::vector<const DeclContext *> &targetContextVector) {
  if (limitToPatternVector.empty()) {
    return true;
  }
  size_t currentTargetIndex = 0;
  for (size_t i = 0; i < limitToPatternVector.size(); ++i) {
    if (!findMatch(currentTargetIndex, targetContextVector,
                   limitToPatternVector[i])) {
      return false;
    }
  }
  for (; currentTargetIndex < targetContextVector.size();
       ++currentTargetIndex) {
    auto currentContext = targetContextVector[currentTargetIndex];
    if (auto currentNamespace = dyn_cast<NamespaceDecl>(currentContext)) {
      if (!isTransparentNamespace(currentNamespace)) {
        return false;
      }
    } else {
      return true;
    }
  }
  return true;
}

IdentifierInfo *UseExplicitNamespacesCheck::getTypeNestedNameIdentfierRecursive(
    const Type *type, std::string &nestedTypeInfo) {
  switch (type->getTypeClass()) {
  case Type::Enum:
  case Type::Record:
    nestedTypeInfo += " TagType";
    return type->castAs<TagType>()->getDecl()->getIdentifier();
  case Type::TemplateSpecialization: {
    nestedTypeInfo += " TemplateSpecializationType";
    auto templateDecl = type->castAs<TemplateSpecializationType>()
                            ->getTemplateName()
                            .getAsTemplateDecl();
    return templateDecl ? templateDecl->getIdentifier() : nullptr;
  }
  case Type::Typedef:
    nestedTypeInfo += " Typedef";
    return type->castAs<TypedefType>()->getDecl()->getIdentifier();
  case Type::Elaborated:
    nestedTypeInfo += " Elaborated";
    return getTypeNestedNameIdentfierRecursive(
        type->castAs<ElaboratedType>()->getNamedType().getTypePtr(),
        nestedTypeInfo);
  case Type::SubstTemplateTypeParm:
    nestedTypeInfo += " SubstTemplateTypeParm";
    return nullptr;
  case Type::Using:
    nestedTypeInfo += " Using";
    return nullptr;
  case Type::InjectedClassName:
    nestedTypeInfo += " InjectedClassName";
    return nullptr;
  default:
    if (diagnosticLevel >= 3) {
      diag(std::string("unexpected type in nested name identifier ") +
           type->getTypeClassName());
    }
    return nullptr;
  }
}

std::string
getNestNamespaceSpecifierKindString(NestedNameSpecifier::SpecifierKind kind) {
  switch (kind) {
  case NestedNameSpecifier::Identifier:
    return "Identifier";
  case NestedNameSpecifier::Namespace:
    return "Namespace ";
  case NestedNameSpecifier::NamespaceAlias:
    return "Namespace Alias";
  case NestedNameSpecifier::TypeSpec:
    return "TypeSpec";
  case NestedNameSpecifier::TypeSpecWithTemplate:
    return "TypeSpecWithTemplate";
  case NestedNameSpecifier::Global:
    return "Global";
  case NestedNameSpecifier::Super:
    return "Super";
  default:
    return "unknown";
  }
}

class QualifierScopeIdentifier {
public:
  QualifierScopeIdentifier(NestedNameSpecifierLoc nestedName,
                           const IdentifierInfo *identifier,
                           const NamespaceDecl *namespaceDecl,
                           bool blocksChange)
      : _nestedName(nestedName), _identifier(identifier),
        _namespaceDecl(namespaceDecl), _blocksChange(blocksChange) {}

  NestedNameSpecifierLoc getNestedName() const { return _nestedName; }

  const IdentifierInfo *getIdentifier() const { return _identifier; }

  bool isNamespace() const { return _namespaceDecl; }

  const NamespaceDecl *getNamespaceDecl() const { return _namespaceDecl; }

  bool getBlocksChange() const { return _blocksChange; }

private:
  NestedNameSpecifierLoc _nestedName;
  const IdentifierInfo *_identifier = nullptr;
  const NamespaceDecl *_namespaceDecl = nullptr;
  bool _blocksChange = false;
};

std::vector<QualifierScopeIdentifier>
getQualifierVector(UseExplicitNamespacesCheck *thisCheck,
                   NestedNameSpecifierLoc nestedName) {
  std::vector<QualifierScopeIdentifier> qualifierVector;
  auto addQualifierNamespace =
      [&qualifierVector](NestedNameSpecifierLoc current,
                         const DeclContext *context) {
        auto namespaceDecl = cast<NamespaceDecl>(context);
        qualifierVector.emplace_back(current, namespaceDecl->getIdentifier(),
                                     namespaceDecl, false);
      };
  for (auto current = nestedName; current.hasQualifier();
       current = current.getPrefix()) {
    auto specifier = current.getNestedNameSpecifier();
    switch (specifier->getKind()) {
    case NestedNameSpecifier::NamespaceAlias: {
      for (const DeclContext *context =
               specifier->getAsNamespaceAlias()->getNamespace();
           context; context = context->getParent()) {
        if (context->isNamespace()) {
          addQualifierNamespace(current, context);
        }
      }
      break;
    }
    case NestedNameSpecifier::Namespace:
      addQualifierNamespace(current, specifier->getAsNamespace());
      break;
    case NestedNameSpecifier::Identifier:
      qualifierVector.emplace_back(current, specifier->getAsIdentifier(),
                                   nullptr, false);
      break;
    case NestedNameSpecifier::TypeSpec: {
      std::string ignoreNested;
      auto identifier = thisCheck->getTypeNestedNameIdentfierRecursive(
          specifier->getAsType(), ignoreNested);
      qualifierVector.emplace_back(current, identifier, nullptr,
                                   identifier == nullptr);
      break;
    }
    default:
      qualifierVector.emplace_back(current, nullptr, nullptr, true);
      break;
    }
  }
  std::reverse(qualifierVector.begin(), qualifierVector.end());
  return qualifierVector;
}

std::string makeNestedNameSpecifierLocString(
    UseExplicitNamespacesCheck *thisCheck,
    std::vector<QualifierScopeIdentifier> &qualifierVector) {
  std::ostringstream out;
  std::for_each(
      qualifierVector.begin(), qualifierVector.end(),
      [&out, thisCheck](const QualifierScopeIdentifier &qualifier) {
        auto specifier = qualifier.getNestedName().getNestedNameSpecifier();
        auto kind = specifier->getKind();
        out << "\t\t" << getNestNamespaceSpecifierKindString(kind) << " "
            << getIdentifierString(qualifier.getIdentifier()).str();
        if (kind == NestedNameSpecifier::TypeSpec) {
          std::string info;
          thisCheck->getTypeNestedNameIdentfierRecursive(specifier->getAsType(),
                                                         info);
          out << " - type info" << info;
        }
        if (qualifier.getBlocksChange()) {
          out << " (blocks change)";
        }
        out << "\n";
      });
  return out.str();
}

std::string
makeDeclContextVectorString(std::vector<const DeclContext *> &contextVector) {
  if (contextVector.empty()) {
    return "\t\tglobal\n";
  }
  std::stringstream out;
  std::for_each(contextVector.begin(), contextVector.end(),
                [&out](const DeclContext *context) {
                  switch (context->getDeclKind()) {
                  case Decl::TranslationUnit:
                    break;
                  case Decl::Namespace:
                  case Decl::CXXMethod:
                  case Decl::CXXRecord:
                  case Decl::Function:
                  case Decl::ClassTemplateSpecialization:
                    out << "\t\t" << context->getDeclKindName();
                    if (const auto *namedDecl = dyn_cast<NamedDecl>(context)) {
                      out << " name " << namedDecl->getNameAsString();
                    }
                    out << "\n";
                    break;
                  default:
                    out << "\t\tunknown decl context kind "
                        << context->getDeclKindName() << "\n";
                    break;
                  }
                });
  return out.str();
}

std::vector<const DeclContext *>
getDeclContextVector(const DeclContext *context) {
  bool namespacesStarted = false;
  std::vector<const DeclContext *> declContextVector;
  for (; context; context = context->getParent()) {
    auto kind = context->getDeclKind();
    switch (kind) {
    case Decl::TranslationUnit:
    case Decl::LinkageSpec:
      break;
    case Decl::Namespace:
      namespacesStarted = true;
      [[fallthrough]];
    default:
      if (namespacesStarted && kind != Decl::Namespace) {
        std::cout << "processing namepsace vector after namespace "
                     "we have "
                  << context->getDeclKindName() << std::endl;
      }
      declContextVector.push_back(context);
    }
  }
  std::reverse(declContextVector.begin(), declContextVector.end());
  return declContextVector;
}

const DeclContext *findDeclContextRecursive(ParentMapContext &parentMapContext,
                                            const DynTypedNode &node) {
  auto decl = node.get<Decl>();
  const DeclContext *nodeAsContext =
      decl ? dyn_cast<DeclContext>(decl) : nullptr;
  if (nodeAsContext) {
    switch (nodeAsContext->getDeclKind()) {
    case Decl::TranslationUnit:
    case Decl::Namespace:
      return nodeAsContext;
    default:
      break;
    }
  }
  auto parents = parentMapContext.getParents(node);
  for (size_t i = 0; i < parents.size(); ++i) {
    auto recurseContext =
        findDeclContextRecursive(parentMapContext, parents[i]);
    if (recurseContext) {
      return (nodeAsContext && nodeAsContext->getParent() == recurseContext)
                 ? nodeAsContext
                 : recurseContext;
    }
  }
  return nullptr;
}

bool isIdentifierCharacter(char character) {
  return (character >= 'a' && character <= 'z') ||
         (character >= 'A' && character <= 'Z') ||
         (character >= '0' && character <= '9') || character == '_';
}

size_t referenceLocationMatchLength(
    const std::vector<const DeclContext *> &targetContextVector,
    const std::vector<const DeclContext *> &referenceContextVector,
    size_t missingStart) {
  size_t minVectorSize =
      std::min(targetContextVector.size(), referenceContextVector.size());
  size_t attemptSize = std::min(minVectorSize, missingStart);
  for (size_t i = 0; i < attemptSize; ++i) {
    if (!targetContextVector[i]->Equals(referenceContextVector[i])) {
      return i;
    }
  }
  return attemptSize;
}

bool explicitContextBlocksChange(
    const std::vector<QualifierScopeIdentifier> &explicitQualifierVector) {
  return std::find_if(explicitQualifierVector.begin(),
                      explicitQualifierVector.end(),
                      [](const QualifierScopeIdentifier &qualifier) {
                        return qualifier.getBlocksChange();
                      }) != explicitQualifierVector.end();
}

bool namespacesMatch(
    const std::vector<const DeclContext *> &targetContextVector, size_t offset,
    const std::vector<QualifierScopeIdentifier> &explicitQualifierVector,
    size_t explicitNamespaceCount) {
  size_t targetIndex = offset;
  auto matchNamespace =
      [&targetContextVector,
       &targetIndex](const NamespaceDecl *explicitNamespaceDecl) -> bool {
    while (targetIndex < targetContextVector.size()) {
      auto targetDecl = targetContextVector[targetIndex];
      ++targetIndex;
      if (targetDecl->Equals(explicitNamespaceDecl)) {
        return true;
      }
      if (!cast<NamespaceDecl>(targetDecl)->isInline()) {
        return false;
      }
    }
    return false;
  };
  for (size_t i = 0; i < explicitNamespaceCount; ++i) {
    if (!matchNamespace(explicitQualifierVector[i].getNamespaceDecl())) {
      return false;
    }
  }
  return true;
}

bool doesContextIdMatch(const IdentifierInfo *contextId,
                        const QualifierScopeIdentifier &qualifier) {
  auto nestedId = qualifier.getIdentifier();
  return contextId && nestedId &&
         contextId->getName().equals(nestedId->getName());
}

bool findContextIdMatch(
    const IdentifierInfo *contextId,
    const std::vector<QualifierScopeIdentifier> &qualifierVector,
    size_t &nextIndex) {
  for (size_t i = nextIndex; i < qualifierVector.size(); i = nextIndex) {
    ++nextIndex;
    if (doesContextIdMatch(contextId, qualifierVector[i])) {
      return true;
    }
  }
  return false;
}

bool hasNonTransparentNamespace(
    const std::vector<const DeclContext *> &targetContextVector,
    size_t startIndex, size_t endIndex) {
  for (size_t i = startIndex; i < endIndex; ++i) {
    auto decl = cast<NamespaceDecl>(targetContextVector[i]);
    if (!isTransparentNamespace(decl)) {
      return true;
    }
  }
  return false;
}

std::string makeMissingTextString(
    const std::vector<const DeclContext *> &targetContextVector,
    size_t startIndex, size_t endIndex) {
  std::string missingText;
  for (size_t i = startIndex; i < endIndex; ++i) {
    auto decl = cast<NamespaceDecl>(targetContextVector[i]);
    if (!isTransparentNamespace(decl)) {
      missingText += decl->getNameAsString() + "::";
    }
  }
  return missingText;
}

void UseExplicitNamespacesCheck::diagOut(const SourceLocation &sourcePosition,
                                         const std::string &message) {
  if (!sourcePosition.isValid()) {
    diag(message);
  } else {
    diag(sourcePosition, message);
  }
}

void UseExplicitNamespacesCheck::processTransform(
    NestedNameSpecifierLoc nestedName, const SourceLocation &sourcePosition,
    const NamedDecl *target, const DeclContext *referenceContext,
    bool usingShadow, const std::string &context) {
  auto targetContextVector = getDeclContextVector(target->getDeclContext());
  auto referenceContextVector = getDeclContextVector(referenceContext);
  auto explicitQualifierVector = getQualifierVector(this, nestedName);
  std::string infoDump;
  if (diagnosticLevel) {
    infoDump = "found identifier " + target->getNameAsString() + "\n" +
               "\tcall sequence\n" + context + "\treference context\n" +
               makeDeclContextVectorString(referenceContextVector) +
               "\ttarget context\n" +
               makeDeclContextVectorString(targetContextVector) +
               "\texplicit context\n" +
               makeNestedNameSpecifierLocString(this, explicitQualifierVector) +
               "\tusing shadow\n\t\t" + trueFalseString(usingShadow).str() +
               "\n";
  }
  if (!matchesNamespaceLimits(targetContextVector)) {
    if (diagnosticLevel >= 2) {
      diagOut(sourcePosition, "does not match namespace limits" + infoDump);
    }
    return;
  }
  if (explicitContextBlocksChange(explicitQualifierVector)) {
    if (diagnosticLevel >= 2) {
      diagOut(sourcePosition, "explicit context blocks change" + infoDump);
    }
    return;
  }
  if (onlyExpandUsingNamespace && usingShadow) {
    if (diagnosticLevel >= 2) {
      diagOut(sourcePosition,
              "onlyExpandingUsingNamespace is true blocks change" + infoDump);
    }
    return;
  }
  size_t targetNamespaceCount = std::count_if(
      targetContextVector.begin(), targetContextVector.end(),
      [](const DeclContext *context) { return context->isNamespace(); });
  size_t explicitNamespaceCount = std::count_if(
      explicitQualifierVector.begin(), explicitQualifierVector.end(),
      [](QualifierScopeIdentifier qualifier) {
        return qualifier.isNamespace();
      });
  size_t nextExplicitQualifierIndex = explicitNamespaceCount;
  for (size_t i = targetNamespaceCount; i < targetContextVector.size(); ++i) {
    auto context = targetContextVector[i];
    if (!context->isTransparentContext()) {
      if (const auto *namedDecl = dyn_cast<NamedDecl>(context)) {
        auto contextId = namedDecl->getIdentifier();
        if (!findContextIdMatch(contextId, explicitQualifierVector,
                                nextExplicitQualifierIndex)) {
          // a required explicit type context was missing, there
          // are a number of situations where this can happen like
          // when a derived class uses a typedef or innter type
          // from its parent class.  None of these situations
          // should result in namespaces being added.
          if (diagnosticLevel >= 2) {
            std::stringstream out;
            out << "failed to find context " << context->getDeclKindName()
                << " named " << getIdentifierString(contextId).str()
                << " target decl kind is " << target->getDeclKindName() << "\n";
            diagOut(sourcePosition, out.str() + infoDump);
          }
          return;
        }
      } else {
        if (diagnosticLevel >= 3) {
          diagOut(sourcePosition, "unnamed non-transparent DeclContext");
        }
      }
    }
  }
  if (explicitNamespaceCount > targetNamespaceCount) {
    if (diagnosticLevel >= 2) {
      diagOut(sourcePosition, "explicit namespace count exceeds "
                              "target namespace count" +
                                  infoDump);
    }
    return;
  }
  size_t possibleShifts = targetNamespaceCount - explicitNamespaceCount;
  for (size_t i = 0; i <= possibleShifts; ++i) {
    if (namespacesMatch(targetContextVector, i, explicitQualifierVector,
                        explicitNamespaceCount)) {
      bool modified = false;
      size_t beforeExplicitNamespace = targetNamespaceCount;
      size_t afterExplicitNamespace = 0;
      size_t afterExplicitStart = 0;
      if (explicitNamespaceCount) {
        beforeExplicitNamespace = i;
        afterExplicitNamespace =
            targetNamespaceCount - explicitNamespaceCount - i;
        afterExplicitStart = targetNamespaceCount - afterExplicitNamespace;
      }
      if (beforeExplicitNamespace) {
        auto neededStart = referenceLocationMatchLength(
            targetContextVector, referenceContextVector,
            beforeExplicitNamespace);
        if (neededStart < beforeExplicitNamespace &&
            hasNonTransparentNamespace(targetContextVector, neededStart,
                                       beforeExplicitNamespace)) {
          auto modifyPosition =
              (explicitQualifierVector.empty())
                  ? sourcePosition
                  : explicitQualifierVector[0].getNestedName().getBeginLoc();
          if (modifyPosition.isValid()) {
            auto missing = makeMissingTextString(
                targetContextVector, neededStart, beforeExplicitNamespace);

            diag(modifyPosition,
                 "Missing namespace qualifiers " + missing + "\n" + infoDump)
                << "Add the qualifying namespaces"
                << FixItHint::CreateInsertion(modifyPosition, missing);
            modified = true;
          }
        }
      }
      if (afterExplicitNamespace &&
          hasNonTransparentNamespace(targetContextVector, afterExplicitStart,
                                     targetNamespaceCount)) {
        auto modifyPosition =
            (explicitQualifierVector.size() > explicitNamespaceCount)
                ? explicitQualifierVector[explicitNamespaceCount]
                      .getNestedName()
                      .getBeginLoc()
                : sourcePosition;
        if (modifyPosition.isValid()) {
          auto missing = makeMissingTextString(
              targetContextVector, afterExplicitStart, targetNamespaceCount);
          diag(modifyPosition, "Missing namespace qualifiers " + missing +
                                   " after explicit namespace\n" + infoDump)
              << "Add the qualifying namespaces"
              << FixItHint::CreateInsertion(modifyPosition, missing);
          modified = true;
        }
      }
      if (!modified && diagnosticLevel >= 2) {
        diagOut(sourcePosition, infoDump);
      }
      return;
    }
  }
  if (diagnosticLevel >= 2) {
    diagOut(sourcePosition, infoDump);
  }
}

void UseExplicitNamespacesCheck::processTypePiecesRecursive(
    NestedNameSpecifierLoc nestedName, const TypeLoc &typeLoc,
    const DeclContext *declContext, const std::string &context) {
  std::stringstream out;
  switch (typeLoc.getTypeLocClass()) {
  case TypeLoc::Qualified:
    processTypePiecesRecursive(nestedName, typeLoc.getUnqualifiedLoc(),
                               declContext,
                               context + "\t\tunpack qualification\n");
    return;
  case TypeLoc::Enum:
  case TypeLoc::Record:
    processTransform(nestedName, typeLoc.getBeginLoc(),
                     typeLoc.castAs<TagTypeLoc>().getDecl(), declContext, false,
                     context + "\t\ttag type name\n");
    return;
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    processTypePiecesRecursive(
        nestedName, typeLoc.castAs<ReferenceTypeLoc>().getPointeeLoc(),
        declContext, context + "\t\tunpack reference\n");
    return;
  case TypeLoc::Pointer:
    processTypePiecesRecursive(nestedName,
                               typeLoc.castAs<PointerTypeLoc>().getPointeeLoc(),
                               declContext, context + "\t\tunpack pointer\n");
    return;
  case TypeLoc::Elaborated: {
    auto elaboratedTypeLoc = typeLoc.castAs<ElaboratedTypeLoc>();
    processTypePiecesRecursive(elaboratedTypeLoc.getQualifierLoc(),
                               elaboratedTypeLoc.getNamedTypeLoc(), declContext,
                               context +
                                   "\t\tunpack elaborated type keyword\n");
    return;
  }
  case TypeLoc::TemplateSpecialization: {
    auto templateLoc = typeLoc.castAs<TemplateSpecializationTypeLoc>();
    auto templateName = templateLoc.getTypePtr()->getTemplateName();
    if (auto target = templateName.getAsTemplateDecl()) {
      processTransform(nestedName, typeLoc.getBeginLoc(), target, declContext,
                       templateName.getAsUsingShadowDecl(),
                       context + "\t\ttemplate name\n");
    }
    for (unsigned i = 0; i < templateLoc.getNumArgs(); ++i) {
      std::string templateArgContext =
          context + "\t\ttemplate[" + std::to_string(i) + "]\n";
      auto typeSourceInfo = templateLoc.getArgLoc(i).getTypeSourceInfo();
      if (typeSourceInfo) {
        processTypePiecesRecursive(NestedNameSpecifierLoc(),
                                   typeSourceInfo->getTypeLoc(), declContext,
                                   templateArgContext);
      } else if (diagnosticLevel >= 3) {
        diagOut(templateLoc.getBeginLoc(),
                "missing template type source info in "
                "context " +
                    templateArgContext);
      }
    }
    return;
  }
  case TypeLoc::ConstantArray:
  case TypeLoc::IncompleteArray:
  case TypeLoc::VariableArray:
    processTypePiecesRecursive(
        nestedName, typeLoc.castAs<ArrayTypeLoc>().getElementLoc(), declContext,
        context + "\t\tunpack array type\n");
    return;
  case TypeLoc::Typedef:
    processTransform(nestedName, typeLoc.getBeginLoc(),
                     typeLoc.castAs<TypedefTypeLoc>().getTypedefNameDecl(),
                     declContext, false, context + "\t\ttypedef type name\n");
    return;
  case TypeLoc::FunctionProto: {
    auto returnTypeLoc = typeLoc.castAs<FunctionProtoTypeLoc>().getReturnLoc();
    if (returnTypeLoc.getTypeLocClass() == TypeLoc::Pointer) {
      auto pointerTypeLoc =
          returnTypeLoc.castAs<PointerTypeLoc>().getPointeeLoc();
      if (pointerTypeLoc.getTypeLocClass() == TypeLoc::FunctionProto) {
        // lambda's do something strange that shows up as
        // FunctionProto, Pointer, FunctionProto and this screws
        // up the source location for things inside, this is a
        // hack that we just skip this
        return;
      }
    }
    processTypePiecesRecursive(nestedName, returnTypeLoc, declContext,
                               context + "\t\tfunction prototype return\n");
    return;
  }
  case TypeLoc::Using:
    processTransform(
        nestedName, typeLoc.getBeginLoc(),
        typeLoc.castAs<UsingTypeLoc>().getFoundDecl()->getTargetDecl(),
        declContext, true, context + "\t\tusing type name\n");
    return;
  case TypeLoc::Builtin:
  case TypeLoc::Auto:
  case TypeLoc::SubstTemplateTypeParm:
  case TypeLoc::TemplateTypeParm:
  case TypeLoc::Paren:
  case TypeLoc::TypeOf:
  case TypeLoc::TypeOfExpr:
  case TypeLoc::DependentName:
  case TypeLoc::Decltype:
  case TypeLoc::PackExpansion:
  case TypeLoc::InjectedClassName:
  case TypeLoc::DependentSizedArray:
  case TypeLoc::MemberPointer:
  case TypeLoc::Complex:
  case TypeLoc::DependentTemplateSpecialization:
  case TypeLoc::UnaryTransform:
    return;
  case TypeLoc::Atomic:
  case TypeLoc::BitInt:
  case TypeLoc::BlockPointer:
  case TypeLoc::DeducedTemplateSpecialization:
  case TypeLoc::FunctionNoProto:
  case TypeLoc::ConstantMatrix:
  case TypeLoc::ObjCObjectPointer:
  case TypeLoc::ObjCObject:
  case TypeLoc::ObjCInterface:
  case TypeLoc::Pipe:
  case TypeLoc::Vector:
  case TypeLoc::ExtVector:
    if (diagnosticLevel >= 3) {
      out << "found unexpected type class "
          << typeLoc.getTypePtr()->getTypeClassName() << "\n"
          << context;
      diagOut(typeLoc.getBeginLoc(), out.str());
    }
    return;
  default:
    if (diagnosticLevel >= 3) {
      out << "unknown type class " << typeLoc.getTypePtr()->getTypeClassName()
          << "\n"
          << context;
      diagOut(typeLoc.getBeginLoc(), out.str());
    }
    return;
  }
}

void UseExplicitNamespacesCheck::processTypePieces(
    TypeSourceInfo *typeSourceInfo, const DeclContext *declContext,
    const std::string &context) {
  if (!typeSourceInfo) {
    if (diagnosticLevel >= 3) {
      diag("missing type source info in context " + context);
    }
    return;
  }
  processTypePiecesRecursive(NestedNameSpecifierLoc(),
                             typeSourceInfo->getTypeLoc(), declContext,
                             "\t\t" + context + "\n");
}

void UseExplicitNamespacesCheck::check(const MatchFinder::MatchResult &Result) {
  if (auto expressionRef = Result.Nodes.getNodeAs<DeclRefExpr>("DeclRefExpr")) {
    auto matchedNode = DynTypedNode::create(*expressionRef);
    auto currentContext = findDeclContextRecursive(
        Result.Context->getParentMapContext(), matchedNode);
    auto matchContext = getMatchContext("DeclRefExpr", matchedNode).str();
    auto foundDecl = expressionRef->getFoundDecl();
    auto foundDeclName = foundDecl->getNameAsString();
    bool notOperator = foundDeclName.size() <= 8 ||
                       foundDeclName.substr(0, 8) != "operator" ||
                       isIdentifierCharacter(foundDeclName[8]);
    if (notOperator) {
      processTransform(expressionRef->getQualifierLoc(),
                       expressionRef->getLocation(), foundDecl, currentContext,
                       false, "\t\t" + matchContext + "\n");
    }
    auto arguments = expressionRef->template_arguments();
    for (size_t i = 0; i < arguments.size(); ++i) {
      processTypePieces(arguments[i].getTypeSourceInfo(), currentContext,
                        " " + matchContext + " template[" + std::to_string(i) +
                            "]");
    }
  } else if (auto newExpr = Result.Nodes.getNodeAs<CXXNewExpr>("CXXNewExpr")) {
    auto matchedNode = DynTypedNode::create(*newExpr);
    auto currentContext = findDeclContextRecursive(
        Result.Context->getParentMapContext(), matchedNode);
    processTypePieces(newExpr->getAllocatedTypeSourceInfo(), currentContext,
                      getMatchContext("CXXNewExpr", matchedNode).str());
  } else if (auto tempObjectExpr =
                 Result.Nodes.getNodeAs<CXXTemporaryObjectExpr>(
                     "CXXTemporaryObjectExpr")) {
    auto matchedNode = DynTypedNode::create(*tempObjectExpr);
    auto currentContext = findDeclContextRecursive(
        Result.Context->getParentMapContext(), matchedNode);
    processTypePieces(
        tempObjectExpr->getTypeSourceInfo(), currentContext,
        getMatchContext("CXXTemporaryObjectExpr", matchedNode).str());
  } else if (auto declaratorDecl =
                 Result.Nodes.getNodeAs<DeclaratorDecl>("DeclaratorDecl")) {
    auto matchedNode = DynTypedNode::create(*declaratorDecl);
    auto currentContext = findDeclContextRecursive(
        Result.Context->getParentMapContext(), matchedNode);
    processTypePieces(declaratorDecl->getTypeSourceInfo(), currentContext,
                      getMatchContext("DeclaratorDecl", matchedNode).str());
  } else if (auto typedefNameDecl =
                 Result.Nodes.getNodeAs<TypedefNameDecl>("TypedefNameDecl")) {
    auto matchedNode = DynTypedNode::create(*typedefNameDecl);
    auto currentContext = findDeclContextRecursive(
        Result.Context->getParentMapContext(), matchedNode);
    processTypePieces(typedefNameDecl->getTypeSourceInfo(), currentContext,
                      getMatchContext("TypedefNameDecl", matchedNode).str());
  }
}

} // namespace clang::tidy::readability
