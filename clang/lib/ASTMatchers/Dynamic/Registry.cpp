//===--- Registry.cpp - Matcher registry -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------===//
///
/// \file
/// \brief Registry map populated at static initialization time.
///
//===------------------------------------------------------------===//

#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "Marshallers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include <set>
#include <utility>

using namespace clang::ast_type_traits;

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using internal::MatcherDescriptor;

typedef llvm::StringMap<const MatcherDescriptor *> ConstructorMap;
class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName, MatcherDescriptor *Callback);
  ConstructorMap Constructors;
};

void RegistryMaps::registerMatcher(StringRef MatcherName,
                                   MatcherDescriptor *Callback) {
  assert(Constructors.find(MatcherName) == Constructors.end());
  Constructors[MatcherName] = Callback;
}

#define REGISTER_MATCHER(name)                                                 \
  registerMatcher(#name, internal::makeMatcherAutoMarshall(                    \
                             ::clang::ast_matchers::name, #name));

#define SPECIFIC_MATCHER_OVERLOAD(name, Id)                                    \
  static_cast< ::clang::ast_matchers::name##_Type##Id>(                        \
      ::clang::ast_matchers::name)

#define REGISTER_OVERLOADED_2(name)                                            \
  do {                                                                         \
    MatcherDescriptor *Callbacks[] = {                                         \
      internal::makeMatcherAutoMarshall(SPECIFIC_MATCHER_OVERLOAD(name, 0),    \
                                        #name),                                \
      internal::makeMatcherAutoMarshall(SPECIFIC_MATCHER_OVERLOAD(name, 1),    \
                                        #name)                                 \
    };                                                                         \
    registerMatcher(#name,                                                     \
                    new internal::OverloadedMatcherDescriptor(Callbacks));     \
  } while (0)

/// \brief Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  // TODO: Here is the list of the missing matchers, grouped by reason.
  //
  // Need Variant/Parser fixes:
  // ofKind
  //
  // Polymorphic + argument overload:
  // findAll
  //
  // Other:
  // equals
  // equalsNode

  REGISTER_OVERLOADED_2(callee);
  REGISTER_OVERLOADED_2(hasPrefix);
  REGISTER_OVERLOADED_2(hasType);
  REGISTER_OVERLOADED_2(isDerivedFrom);
  REGISTER_OVERLOADED_2(isSameOrDerivedFrom);
  REGISTER_OVERLOADED_2(loc);
  REGISTER_OVERLOADED_2(pointsTo);
  REGISTER_OVERLOADED_2(references);
  REGISTER_OVERLOADED_2(thisPointerType);

  REGISTER_MATCHER(accessSpecDecl);
  REGISTER_MATCHER(alignOfExpr);
  REGISTER_MATCHER(allOf);
  REGISTER_MATCHER(anyOf);
  REGISTER_MATCHER(anything);
  REGISTER_MATCHER(argumentCountIs);
  REGISTER_MATCHER(arraySubscriptExpr);
  REGISTER_MATCHER(arrayType);
  REGISTER_MATCHER(asmStmt);
  REGISTER_MATCHER(asString);
  REGISTER_MATCHER(atomicType);
  REGISTER_MATCHER(autoType);
  REGISTER_MATCHER(binaryOperator);
  REGISTER_MATCHER(blockPointerType);
  REGISTER_MATCHER(breakStmt);
  REGISTER_MATCHER(builtinType);
  REGISTER_MATCHER(callExpr);
  REGISTER_MATCHER(caseStmt);
  REGISTER_MATCHER(castExpr);
  REGISTER_MATCHER(characterLiteral);
  REGISTER_MATCHER(classTemplateDecl);
  REGISTER_MATCHER(classTemplateSpecializationDecl);
  REGISTER_MATCHER(complexType);
  REGISTER_MATCHER(compoundLiteralExpr);
  REGISTER_MATCHER(compoundStmt);
  REGISTER_MATCHER(conditionalOperator);
  REGISTER_MATCHER(constantArrayType);
  REGISTER_MATCHER(containsDeclaration);
  REGISTER_MATCHER(continueStmt);
  REGISTER_MATCHER(cStyleCastExpr);
  REGISTER_MATCHER(cudaKernelCallExpr);
  REGISTER_MATCHER(cxxBindTemporaryExpr);
  REGISTER_MATCHER(cxxBoolLiteral);
  REGISTER_MATCHER(cxxCatchStmt);
  REGISTER_MATCHER(cxxConstCastExpr);
  REGISTER_MATCHER(cxxConstructExpr);
  REGISTER_MATCHER(cxxConstructorDecl);
  REGISTER_MATCHER(cxxConversionDecl);
  REGISTER_MATCHER(cxxCtorInitializer);
  REGISTER_MATCHER(cxxDefaultArgExpr);
  REGISTER_MATCHER(cxxDeleteExpr);
  REGISTER_MATCHER(cxxDestructorDecl);
  REGISTER_MATCHER(cxxDynamicCastExpr);
  REGISTER_MATCHER(cxxForRangeStmt);
  REGISTER_MATCHER(cxxFunctionalCastExpr);
  REGISTER_MATCHER(cxxMemberCallExpr);
  REGISTER_MATCHER(cxxMethodDecl);
  REGISTER_MATCHER(cxxNewExpr);
  REGISTER_MATCHER(cxxNullPtrLiteralExpr);
  REGISTER_MATCHER(cxxOperatorCallExpr);
  REGISTER_MATCHER(cxxRecordDecl);
  REGISTER_MATCHER(cxxReinterpretCastExpr);
  REGISTER_MATCHER(cxxStaticCastExpr);
  REGISTER_MATCHER(cxxTemporaryObjectExpr);
  REGISTER_MATCHER(cxxThisExpr);
  REGISTER_MATCHER(cxxThrowExpr);
  REGISTER_MATCHER(cxxTryStmt);
  REGISTER_MATCHER(cxxUnresolvedConstructExpr);
  REGISTER_MATCHER(decayedType);
  REGISTER_MATCHER(decl);
  REGISTER_MATCHER(declaratorDecl);
  REGISTER_MATCHER(declCountIs);
  REGISTER_MATCHER(declRefExpr);
  REGISTER_MATCHER(declStmt);
  REGISTER_MATCHER(defaultStmt);
  REGISTER_MATCHER(dependentSizedArrayType);
  REGISTER_MATCHER(doStmt);
  REGISTER_MATCHER(eachOf);
  REGISTER_MATCHER(elaboratedType);
  REGISTER_MATCHER(enumConstantDecl);
  REGISTER_MATCHER(enumDecl);
  REGISTER_MATCHER(equalsBoundNode);
  REGISTER_MATCHER(equalsIntegralValue);
  REGISTER_MATCHER(explicitCastExpr);
  REGISTER_MATCHER(expr);
  REGISTER_MATCHER(exprWithCleanups);
  REGISTER_MATCHER(fieldDecl);
  REGISTER_MATCHER(floatLiteral);
  REGISTER_MATCHER(forEach);
  REGISTER_MATCHER(forEachConstructorInitializer);
  REGISTER_MATCHER(forEachDescendant);
  REGISTER_MATCHER(forEachSwitchCase);
  REGISTER_MATCHER(forField);
  REGISTER_MATCHER(forStmt);
  REGISTER_MATCHER(friendDecl);
  REGISTER_MATCHER(functionDecl);
  REGISTER_MATCHER(functionTemplateDecl);
  REGISTER_MATCHER(functionType);
  REGISTER_MATCHER(gotoStmt);
  REGISTER_MATCHER(has);
  REGISTER_MATCHER(hasAncestor);
  REGISTER_MATCHER(hasAnyArgument);
  REGISTER_MATCHER(hasAnyConstructorInitializer);
  REGISTER_MATCHER(hasAnyParameter);
  REGISTER_MATCHER(hasAnySubstatement);
  REGISTER_MATCHER(hasAnyTemplateArgument);
  REGISTER_MATCHER(hasAnyUsingShadowDecl);
  REGISTER_MATCHER(hasArgument);
  REGISTER_MATCHER(hasArgumentOfType);
  REGISTER_MATCHER(hasAttr);
  REGISTER_MATCHER(hasAutomaticStorageDuration);
  REGISTER_MATCHER(hasBase);
  REGISTER_MATCHER(hasBody);
  REGISTER_MATCHER(hasCanonicalType);
  REGISTER_MATCHER(hasCaseConstant);
  REGISTER_MATCHER(hasCondition);
  REGISTER_MATCHER(hasConditionVariableStatement);
  REGISTER_MATCHER(hasDecayedType);
  REGISTER_MATCHER(hasDeclaration);
  REGISTER_MATCHER(hasDeclContext);
  REGISTER_MATCHER(hasDeducedType);
  REGISTER_MATCHER(hasDescendant);
  REGISTER_MATCHER(hasDestinationType);
  REGISTER_MATCHER(hasEitherOperand);
  REGISTER_MATCHER(hasElementType);
  REGISTER_MATCHER(hasElse);
  REGISTER_MATCHER(hasFalseExpression);
  REGISTER_MATCHER(hasGlobalStorage);
  REGISTER_MATCHER(hasImplicitDestinationType);
  REGISTER_MATCHER(hasIncrement);
  REGISTER_MATCHER(hasIndex);
  REGISTER_MATCHER(hasInitializer);
  REGISTER_MATCHER(hasKeywordSelector);
  REGISTER_MATCHER(hasLHS);
  REGISTER_MATCHER(hasLocalQualifiers);
  REGISTER_MATCHER(hasLocalStorage);
  REGISTER_MATCHER(hasLoopInit);
  REGISTER_MATCHER(hasLoopVariable);
  REGISTER_MATCHER(hasMethod);
  REGISTER_MATCHER(hasName);
  REGISTER_MATCHER(hasNullSelector);
  REGISTER_MATCHER(hasObjectExpression);
  REGISTER_MATCHER(hasOperatorName);
  REGISTER_MATCHER(hasOverloadedOperatorName);
  REGISTER_MATCHER(hasParameter);
  REGISTER_MATCHER(hasParent);
  REGISTER_MATCHER(hasQualifier);
  REGISTER_MATCHER(hasRangeInit);
  REGISTER_MATCHER(hasReceiverType);
  REGISTER_MATCHER(hasRHS);
  REGISTER_MATCHER(hasSelector);
  REGISTER_MATCHER(hasSingleDecl);
  REGISTER_MATCHER(hasSize);
  REGISTER_MATCHER(hasSizeExpr);
  REGISTER_MATCHER(hasSourceExpression);
  REGISTER_MATCHER(hasStaticStorageDuration);
  REGISTER_MATCHER(hasTargetDecl);
  REGISTER_MATCHER(hasTemplateArgument);
  REGISTER_MATCHER(hasThen);
  REGISTER_MATCHER(hasThreadStorageDuration);
  REGISTER_MATCHER(hasTrueExpression);
  REGISTER_MATCHER(hasTypeLoc);
  REGISTER_MATCHER(hasUnaryOperand);
  REGISTER_MATCHER(hasUnarySelector);
  REGISTER_MATCHER(hasValueType);
  REGISTER_MATCHER(ifStmt);
  REGISTER_MATCHER(ignoringImpCasts);
  REGISTER_MATCHER(ignoringParenCasts);
  REGISTER_MATCHER(ignoringParenImpCasts);
  REGISTER_MATCHER(implicitCastExpr);
  REGISTER_MATCHER(incompleteArrayType);
  REGISTER_MATCHER(initListExpr);
  REGISTER_MATCHER(injectedClassNameType);
  REGISTER_MATCHER(innerType);
  REGISTER_MATCHER(integerLiteral);
  REGISTER_MATCHER(isAnonymous);
  REGISTER_MATCHER(isArrow);
  REGISTER_MATCHER(isBaseInitializer);
  REGISTER_MATCHER(isCatchAll);
  REGISTER_MATCHER(isClass);
  REGISTER_MATCHER(isConst);
  REGISTER_MATCHER(isConstQualified);
  REGISTER_MATCHER(isCopyConstructor);
  REGISTER_MATCHER(isDefaultConstructor);
  REGISTER_MATCHER(isDefinition);
  REGISTER_MATCHER(isDeleted);
  REGISTER_MATCHER(isExceptionVariable);
  REGISTER_MATCHER(isExplicit);
  REGISTER_MATCHER(isExplicitTemplateSpecialization);
  REGISTER_MATCHER(isExpr);
  REGISTER_MATCHER(isExternC);
  REGISTER_MATCHER(isFinal);
  REGISTER_MATCHER(isInline);
  REGISTER_MATCHER(isImplicit);
  REGISTER_MATCHER(isExpansionInFileMatching);
  REGISTER_MATCHER(isExpansionInMainFile);
  REGISTER_MATCHER(isInstantiated);
  REGISTER_MATCHER(isExpansionInSystemHeader);
  REGISTER_MATCHER(isInteger);
  REGISTER_MATCHER(isIntegral);
  REGISTER_MATCHER(isInTemplateInstantiation);
  REGISTER_MATCHER(isListInitialization);
  REGISTER_MATCHER(isMemberInitializer);
  REGISTER_MATCHER(isMoveConstructor);
  REGISTER_MATCHER(isNoThrow);
  REGISTER_MATCHER(isOverride);
  REGISTER_MATCHER(isPrivate);
  REGISTER_MATCHER(isProtected);
  REGISTER_MATCHER(isPublic);
  REGISTER_MATCHER(isPure);
  REGISTER_MATCHER(isStruct);
  REGISTER_MATCHER(isTemplateInstantiation);
  REGISTER_MATCHER(isUnion);
  REGISTER_MATCHER(isVariadic);
  REGISTER_MATCHER(isVirtual);
  REGISTER_MATCHER(isVolatileQualified);
  REGISTER_MATCHER(isWritten);
  REGISTER_MATCHER(labelStmt);
  REGISTER_MATCHER(lambdaExpr);
  REGISTER_MATCHER(lValueReferenceType);
  REGISTER_MATCHER(matchesName);
  REGISTER_MATCHER(matchesSelector);
  REGISTER_MATCHER(materializeTemporaryExpr);
  REGISTER_MATCHER(member);
  REGISTER_MATCHER(memberExpr);
  REGISTER_MATCHER(memberPointerType);
  REGISTER_MATCHER(namedDecl);
  REGISTER_MATCHER(namespaceAliasDecl);
  REGISTER_MATCHER(namespaceDecl);
  REGISTER_MATCHER(namesType);
  REGISTER_MATCHER(nestedNameSpecifier);
  REGISTER_MATCHER(nestedNameSpecifierLoc);
  REGISTER_MATCHER(nullStmt);
  REGISTER_MATCHER(numSelectorArgs);
  REGISTER_MATCHER(ofClass);
  REGISTER_MATCHER(objcInterfaceDecl);
  REGISTER_MATCHER(objcMessageExpr);
  REGISTER_MATCHER(objcObjectPointerType);
  REGISTER_MATCHER(on);
  REGISTER_MATCHER(onImplicitObjectArgument);
  REGISTER_MATCHER(parameterCountIs);
  REGISTER_MATCHER(parenType);
  REGISTER_MATCHER(parmVarDecl);
  REGISTER_MATCHER(pointee);
  REGISTER_MATCHER(pointerType);
  REGISTER_MATCHER(qualType);
  REGISTER_MATCHER(recordDecl);
  REGISTER_MATCHER(recordType);
  REGISTER_MATCHER(referenceType);
  REGISTER_MATCHER(refersToDeclaration);
  REGISTER_MATCHER(refersToIntegralType);
  REGISTER_MATCHER(refersToType);
  REGISTER_MATCHER(returns);
  REGISTER_MATCHER(returnStmt);
  REGISTER_MATCHER(rValueReferenceType);
  REGISTER_MATCHER(sizeOfExpr);
  REGISTER_MATCHER(specifiesNamespace);
  REGISTER_MATCHER(specifiesType);
  REGISTER_MATCHER(specifiesTypeLoc);
  REGISTER_MATCHER(statementCountIs);
  REGISTER_MATCHER(staticAssertDecl);
  REGISTER_MATCHER(stmt);
  REGISTER_MATCHER(stringLiteral);
  REGISTER_MATCHER(substNonTypeTemplateParmExpr);
  REGISTER_MATCHER(substTemplateTypeParmType);
  REGISTER_MATCHER(switchCase);
  REGISTER_MATCHER(switchStmt);
  REGISTER_MATCHER(templateArgument);
  REGISTER_MATCHER(templateArgumentCountIs);
  REGISTER_MATCHER(templateSpecializationType);
  REGISTER_MATCHER(templateTypeParmType);
  REGISTER_MATCHER(throughUsingDecl);
  REGISTER_MATCHER(to);
  REGISTER_MATCHER(translationUnitDecl);
  REGISTER_MATCHER(type);
  REGISTER_MATCHER(typedefDecl);
  REGISTER_MATCHER(typedefType);
  REGISTER_MATCHER(typeLoc);
  REGISTER_MATCHER(unaryExprOrTypeTraitExpr);
  REGISTER_MATCHER(unaryOperator);
  REGISTER_MATCHER(unaryTransformType);
  REGISTER_MATCHER(unless);
  REGISTER_MATCHER(unresolvedUsingTypenameDecl);
  REGISTER_MATCHER(unresolvedUsingValueDecl);
  REGISTER_MATCHER(userDefinedLiteral);
  REGISTER_MATCHER(usingDecl);
  REGISTER_MATCHER(usingDirectiveDecl);
  REGISTER_MATCHER(valueDecl);
  REGISTER_MATCHER(varDecl);
  REGISTER_MATCHER(variableArrayType);
  REGISTER_MATCHER(voidType);
  REGISTER_MATCHER(whileStmt);
  REGISTER_MATCHER(withInitializer);
}

RegistryMaps::~RegistryMaps() {
  llvm::DeleteContainerSeconds(Constructors);
}

static llvm::ManagedStatic<RegistryMaps> RegistryData;

} // anonymous namespace

// static
llvm::Optional<MatcherCtor> Registry::lookupMatcherCtor(StringRef MatcherName) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  return it == RegistryData->constructors().end()
             ? llvm::Optional<MatcherCtor>()
             : it->second;
}

namespace {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::set<ASTNodeKind> &KS) {
  unsigned Count = 0;
  for (std::set<ASTNodeKind>::const_iterator I = KS.begin(), E = KS.end();
       I != E; ++I) {
    if (I != KS.begin())
      OS << "|";
    if (Count++ == 3) {
      OS << "...";
      break;
    }
    OS << *I;
  }
  return OS;
}

}  // namespace

std::vector<ArgKind> Registry::getAcceptedCompletionTypes(
    ArrayRef<std::pair<MatcherCtor, unsigned>> Context) {
  ASTNodeKind InitialTypes[] = {
      ASTNodeKind::getFromNodeKind<Decl>(),
      ASTNodeKind::getFromNodeKind<QualType>(),
      ASTNodeKind::getFromNodeKind<Type>(),
      ASTNodeKind::getFromNodeKind<Stmt>(),
      ASTNodeKind::getFromNodeKind<NestedNameSpecifier>(),
      ASTNodeKind::getFromNodeKind<NestedNameSpecifierLoc>(),
      ASTNodeKind::getFromNodeKind<TypeLoc>()};

  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ArgKind> TypeSet(std::begin(InitialTypes), std::end(InitialTypes));
  for (const auto &CtxEntry : Context) {
    MatcherCtor Ctor = CtxEntry.first;
    unsigned ArgNumber = CtxEntry.second;
    std::vector<ArgKind> NextTypeSet;
    for (const ArgKind &Kind : TypeSet) {
      if (Kind.getArgKind() == Kind.AK_Matcher &&
          Ctor->isConvertibleTo(Kind.getMatcherKind()) &&
          (Ctor->isVariadic() || ArgNumber < Ctor->getNumArgs()))
        Ctor->getArgKinds(Kind.getMatcherKind(), ArgNumber, NextTypeSet);
    }
    TypeSet.clear();
    TypeSet.insert(NextTypeSet.begin(), NextTypeSet.end());
  }
  return std::vector<ArgKind>(TypeSet.begin(), TypeSet.end());
}

std::vector<MatcherCompletion>
Registry::getMatcherCompletions(ArrayRef<ArgKind> AcceptedTypes) {
  std::vector<MatcherCompletion> Completions;

  // Search the registry for acceptable matchers.
  for (const auto &M : RegistryData->constructors()) {
    const auto *Matcher = M.getValue();
    StringRef Name = M.getKey();

    std::set<ASTNodeKind> RetKinds;
    unsigned NumArgs = Matcher->isVariadic() ? 1 : Matcher->getNumArgs();
    bool IsPolymorphic = Matcher->isPolymorphic();
    std::vector<std::vector<ArgKind>> ArgsKinds(NumArgs);
    unsigned MaxSpecificity = 0;
    for (const ArgKind& Kind : AcceptedTypes) {
      if (Kind.getArgKind() != Kind.AK_Matcher)
        continue;
      unsigned Specificity;
      ASTNodeKind LeastDerivedKind;
      if (Matcher->isConvertibleTo(Kind.getMatcherKind(), &Specificity,
                                   &LeastDerivedKind)) {
        if (MaxSpecificity < Specificity)
          MaxSpecificity = Specificity;
        RetKinds.insert(LeastDerivedKind);
        for (unsigned Arg = 0; Arg != NumArgs; ++Arg)
          Matcher->getArgKinds(Kind.getMatcherKind(), Arg, ArgsKinds[Arg]);
        if (IsPolymorphic)
          break;
      }
    }

    if (!RetKinds.empty() && MaxSpecificity > 0) {
      std::string Decl;
      llvm::raw_string_ostream OS(Decl);

      if (IsPolymorphic) {
        OS << "Matcher<T> " << Name << "(Matcher<T>";
      } else {
        OS << "Matcher<" << RetKinds << "> " << Name << "(";
        for (const std::vector<ArgKind> &Arg : ArgsKinds) {
          if (&Arg != &ArgsKinds[0])
            OS << ", ";

          bool FirstArgKind = true;
          std::set<ASTNodeKind> MatcherKinds;
          // Two steps. First all non-matchers, then matchers only.
          for (const ArgKind &AK : Arg) {
            if (AK.getArgKind() == ArgKind::AK_Matcher) {
              MatcherKinds.insert(AK.getMatcherKind());
            } else {
              if (!FirstArgKind) OS << "|";
              FirstArgKind = false;
              OS << AK.asString();
            }
          }
          if (!MatcherKinds.empty()) {
            if (!FirstArgKind) OS << "|";
            OS << "Matcher<" << MatcherKinds << ">";
          }
        }
      }
      if (Matcher->isVariadic())
        OS << "...";
      OS << ")";

      std::string TypedText = Name;
      TypedText += "(";
      if (ArgsKinds.empty())
        TypedText += ")";
      else if (ArgsKinds[0][0].getArgKind() == ArgKind::AK_String)
        TypedText += "\"";

      Completions.emplace_back(TypedText, OS.str(), MaxSpecificity);
    }
  }

  return Completions;
}

// static
VariantMatcher Registry::constructMatcher(MatcherCtor Ctor,
                                          SourceRange NameRange,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) {
  return Ctor->create(NameRange, Args, Error);
}

// static
VariantMatcher Registry::constructBoundMatcher(MatcherCtor Ctor,
                                               SourceRange NameRange,
                                               StringRef BindID,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) {
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  if (Out.isNull()) return Out;

  llvm::Optional<DynTypedMatcher> Result = Out.getSingleMatcher();
  if (Result.hasValue()) {
    llvm::Optional<DynTypedMatcher> Bound = Result->tryBind(BindID);
    if (Bound.hasValue()) {
      return VariantMatcher::SingleMatcher(*Bound);
    }
  }
  Error->addError(NameRange, Error->ET_RegistryNotBindable);
  return VariantMatcher();
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
