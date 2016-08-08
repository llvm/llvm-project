//===--- IdentifierNamingCheck.cpp - clang-tidy ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IdentifierNamingCheck.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

#define DEBUG_TYPE "clang-tidy"

using namespace clang::ast_matchers;

namespace llvm {
/// Specialisation of DenseMapInfo to allow NamingCheckId objects in DenseMaps
template <>
struct DenseMapInfo<
    clang::tidy::readability::IdentifierNamingCheck::NamingCheckId> {
  using NamingCheckId =
      clang::tidy::readability::IdentifierNamingCheck::NamingCheckId;

  static inline NamingCheckId getEmptyKey() {
    return NamingCheckId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-1)),
        "EMPTY");
  }

  static inline NamingCheckId getTombstoneKey() {
    return NamingCheckId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-2)),
        "TOMBSTONE");
  }

  static unsigned getHashValue(NamingCheckId Val) {
    assert(Val != getEmptyKey() && "Cannot hash the empty key!");
    assert(Val != getTombstoneKey() && "Cannot hash the tombstone key!");

    std::hash<NamingCheckId::second_type> SecondHash;
    return Val.first.getRawEncoding() + SecondHash(Val.second);
  }

  static bool isEqual(NamingCheckId LHS, NamingCheckId RHS) {
    if (RHS == getEmptyKey())
      return LHS == getEmptyKey();
    if (RHS == getTombstoneKey())
      return LHS == getTombstoneKey();
    return LHS == RHS;
  }
};
} // namespace llvm

namespace clang {
namespace tidy {
namespace readability {

// clang-format off
#define NAMING_KEYS(m) \
    m(Namespace) \
    m(InlineNamespace) \
    m(EnumConstant) \
    m(ConstexprVariable) \
    m(ConstantMember) \
    m(PrivateMember) \
    m(ProtectedMember) \
    m(PublicMember) \
    m(Member) \
    m(ClassConstant) \
    m(ClassMember) \
    m(GlobalConstant) \
    m(GlobalVariable) \
    m(LocalConstant) \
    m(LocalVariable) \
    m(StaticConstant) \
    m(StaticVariable) \
    m(Constant) \
    m(Variable) \
    m(ConstantParameter) \
    m(ParameterPack) \
    m(Parameter) \
    m(AbstractClass) \
    m(Struct) \
    m(Class) \
    m(Union) \
    m(Enum) \
    m(GlobalFunction) \
    m(ConstexprFunction) \
    m(Function) \
    m(ConstexprMethod) \
    m(VirtualMethod) \
    m(ClassMethod) \
    m(PrivateMethod) \
    m(ProtectedMethod) \
    m(PublicMethod) \
    m(Method) \
    m(Typedef) \
    m(TypeTemplateParameter) \
    m(ValueTemplateParameter) \
    m(TemplateTemplateParameter) \
    m(TemplateParameter) \
    m(TypeAlias) \
    m(MacroDefinition) \

enum StyleKind {
#define ENUMERATE(v) SK_ ## v,
  NAMING_KEYS(ENUMERATE)
#undef ENUMERATE
  SK_Count,
  SK_Invalid
};

static StringRef const StyleNames[] = {
#define STRINGIZE(v) #v,
  NAMING_KEYS(STRINGIZE)
#undef STRINGIZE
};

#undef NAMING_KEYS
// clang-format on

namespace {
/// Callback supplies macros to IdentifierNamingCheck::checkMacro
class IdentifierNamingCheckPPCallbacks : public PPCallbacks {
public:
  IdentifierNamingCheckPPCallbacks(Preprocessor *PP,
                                   IdentifierNamingCheck *Check)
      : PP(PP), Check(Check) {}

  /// MacroDefined calls checkMacro for macros in the main file
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    Check->checkMacro(PP->getSourceManager(), MacroNameTok, MD->getMacroInfo());
  }

  /// MacroExpands calls expandMacro for macros in the main file
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange /*Range*/,
                    const MacroArgs * /*Args*/) override {
    Check->expandMacro(MacroNameTok, MD.getMacroInfo());
  }

private:
  Preprocessor *PP;
  IdentifierNamingCheck *Check;
};
} // namespace

IdentifierNamingCheck::IdentifierNamingCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  auto const fromString = [](StringRef Str) {
    return llvm::StringSwitch<CaseType>(Str)
        .Case("lower_case", CT_LowerCase)
        .Case("UPPER_CASE", CT_UpperCase)
        .Case("camelBack", CT_CamelBack)
        .Case("CamelCase", CT_CamelCase)
        .Case("Camel_Snake_Case", CT_CamelSnakeCase)
        .Case("camel_Snake_Back", CT_CamelSnakeBack)
        .Default(CT_AnyCase);
  };

  for (auto const &Name : StyleNames) {
    NamingStyles.push_back(
        NamingStyle(fromString(Options.get((Name + "Case").str(), "")),
                    Options.get((Name + "Prefix").str(), ""),
                    Options.get((Name + "Suffix").str(), "")));
  }

  IgnoreFailedSplit = Options.get("IgnoreFailedSplit", 0);
}

void IdentifierNamingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  auto const toString = [](CaseType Type) {
    switch (Type) {
    case CT_AnyCase:
      return "aNy_CasE";
    case CT_LowerCase:
      return "lower_case";
    case CT_CamelBack:
      return "camelBack";
    case CT_UpperCase:
      return "UPPER_CASE";
    case CT_CamelCase:
      return "CamelCase";
    case CT_CamelSnakeCase:
      return "Camel_Snake_Case";
    case CT_CamelSnakeBack:
      return "camel_Snake_Back";
    }

    llvm_unreachable("Unknown Case Type");
  };

  for (size_t i = 0; i < SK_Count; ++i) {
    Options.store(Opts, (StyleNames[i] + "Case").str(),
                  toString(NamingStyles[i].Case));
    Options.store(Opts, (StyleNames[i] + "Prefix").str(),
                  NamingStyles[i].Prefix);
    Options.store(Opts, (StyleNames[i] + "Suffix").str(),
                  NamingStyles[i].Suffix);
  }

  Options.store(Opts, "IgnoreFailedSplit", IgnoreFailedSplit);
}

void IdentifierNamingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(namedDecl().bind("decl"), this);
  Finder->addMatcher(usingDecl().bind("using"), this);
  Finder->addMatcher(declRefExpr().bind("declRef"), this);
  Finder->addMatcher(cxxConstructorDecl().bind("classRef"), this);
  Finder->addMatcher(cxxDestructorDecl().bind("classRef"), this);
  Finder->addMatcher(typeLoc().bind("typeLoc"), this);
  Finder->addMatcher(nestedNameSpecifierLoc().bind("nestedNameLoc"), this);
}

void IdentifierNamingCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  Compiler.getPreprocessor().addPPCallbacks(
      llvm::make_unique<IdentifierNamingCheckPPCallbacks>(
          &Compiler.getPreprocessor(), this));
}

static bool matchesStyle(StringRef Name,
                         IdentifierNamingCheck::NamingStyle Style) {
  static llvm::Regex Matchers[] = {
      llvm::Regex("^.*$"),
      llvm::Regex("^[a-z][a-z0-9_]*$"),
      llvm::Regex("^[a-z][a-zA-Z0-9]*$"),
      llvm::Regex("^[A-Z][A-Z0-9_]*$"),
      llvm::Regex("^[A-Z][a-zA-Z0-9]*$"),
      llvm::Regex("^[A-Z]([a-z0-9]*(_[A-Z])?)*"),
      llvm::Regex("^[a-z]([a-z0-9]*(_[A-Z])?)*"),
  };

  bool Matches = true;
  if (Name.startswith(Style.Prefix))
    Name = Name.drop_front(Style.Prefix.size());
  else
    Matches = false;

  if (Name.endswith(Style.Suffix))
    Name = Name.drop_back(Style.Suffix.size());
  else
    Matches = false;

  if (!Matchers[static_cast<size_t>(Style.Case)].match(Name))
    Matches = false;

  return Matches;
}

static std::string fixupWithCase(StringRef Name,
                                 IdentifierNamingCheck::CaseType Case) {
  static llvm::Regex Splitter(
      "([a-z0-9A-Z]*)(_+)|([A-Z]?[a-z0-9]+)([A-Z]|$)|([A-Z]+)([A-Z]|$)");

  SmallVector<StringRef, 8> Substrs;
  Name.split(Substrs, "_", -1, false);

  SmallVector<StringRef, 8> Words;
  for (auto Substr : Substrs) {
    while (!Substr.empty()) {
      SmallVector<StringRef, 8> Groups;
      if (!Splitter.match(Substr, &Groups))
        break;

      if (Groups[2].size() > 0) {
        Words.push_back(Groups[1]);
        Substr = Substr.substr(Groups[0].size());
      } else if (Groups[3].size() > 0) {
        Words.push_back(Groups[3]);
        Substr = Substr.substr(Groups[0].size() - Groups[4].size());
      } else if (Groups[5].size() > 0) {
        Words.push_back(Groups[5]);
        Substr = Substr.substr(Groups[0].size() - Groups[6].size());
      }
    }
  }

  if (Words.empty())
    return Name;

  std::string Fixup;
  switch (Case) {
  case IdentifierNamingCheck::CT_AnyCase:
    Fixup += Name;
    break;

  case IdentifierNamingCheck::CT_LowerCase:
    for (auto const &Word : Words) {
      if (&Word != &Words.front())
        Fixup += "_";
      Fixup += Word.lower();
    }
    break;

  case IdentifierNamingCheck::CT_UpperCase:
    for (auto const &Word : Words) {
      if (&Word != &Words.front())
        Fixup += "_";
      Fixup += Word.upper();
    }
    break;

  case IdentifierNamingCheck::CT_CamelCase:
    for (auto const &Word : Words) {
      Fixup += Word.substr(0, 1).upper();
      Fixup += Word.substr(1).lower();
    }
    break;

  case IdentifierNamingCheck::CT_CamelBack:
    for (auto const &Word : Words) {
      if (&Word == &Words.front()) {
        Fixup += Word.lower();
      } else {
        Fixup += Word.substr(0, 1).upper();
        Fixup += Word.substr(1).lower();
      }
    }
    break;

  case IdentifierNamingCheck::CT_CamelSnakeCase:
    for (auto const &Word : Words) {
      if (&Word != &Words.front())
        Fixup += "_";
      Fixup += Word.substr(0, 1).upper();
      Fixup += Word.substr(1).lower();
    }
    break;

  case IdentifierNamingCheck::CT_CamelSnakeBack:
    for (auto const &Word : Words) {
      if (&Word != &Words.front()) {
        Fixup += "_";
        Fixup += Word.substr(0, 1).upper();
      } else {
        Fixup += Word.substr(0, 1).lower();
      }
      Fixup += Word.substr(1).lower();
    }
    break;
  }

  return Fixup;
}

static std::string fixupWithStyle(StringRef Name,
                                  IdentifierNamingCheck::NamingStyle Style) {
  return Style.Prefix + fixupWithCase(Name, Style.Case) + Style.Suffix;
}

static StyleKind findStyleKind(
    const NamedDecl *D,
    const std::vector<IdentifierNamingCheck::NamingStyle> &NamingStyles) {
  if (isa<TypedefDecl>(D) && NamingStyles[SK_Typedef].isSet())
    return SK_Typedef;

  if (isa<TypeAliasDecl>(D) && NamingStyles[SK_TypeAlias].isSet())
    return SK_TypeAlias;

  if (const auto *Decl = dyn_cast<NamespaceDecl>(D)) {
    if (Decl->isAnonymousNamespace())
      return SK_Invalid;

    if (Decl->isInline() && NamingStyles[SK_InlineNamespace].isSet())
      return SK_InlineNamespace;

    if (NamingStyles[SK_Namespace].isSet())
      return SK_Namespace;
  }

  if (isa<EnumDecl>(D) && NamingStyles[SK_Enum].isSet())
    return SK_Enum;

  if (isa<EnumConstantDecl>(D)) {
    if (NamingStyles[SK_EnumConstant].isSet())
      return SK_EnumConstant;

    if (NamingStyles[SK_Constant].isSet())
      return SK_Constant;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<CXXRecordDecl>(D)) {
    if (Decl->isAnonymousStructOrUnion())
      return SK_Invalid;

    if (Decl->hasDefinition() && Decl->isAbstract() &&
        NamingStyles[SK_AbstractClass].isSet())
      return SK_AbstractClass;

    if (Decl->isStruct() && NamingStyles[SK_Struct].isSet())
      return SK_Struct;

    if (Decl->isStruct() && NamingStyles[SK_Class].isSet())
      return SK_Class;

    if (Decl->isClass() && NamingStyles[SK_Class].isSet())
      return SK_Class;

    if (Decl->isClass() && NamingStyles[SK_Struct].isSet())
      return SK_Struct;

    if (Decl->isUnion() && NamingStyles[SK_Union].isSet())
      return SK_Union;

    if (Decl->isEnum() && NamingStyles[SK_Enum].isSet())
      return SK_Enum;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<FieldDecl>(D)) {
    QualType Type = Decl->getType();

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        NamingStyles[SK_ConstantMember].isSet())
      return SK_ConstantMember;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        NamingStyles[SK_Constant].isSet())
      return SK_Constant;

    if (Decl->getAccess() == AS_private &&
        NamingStyles[SK_PrivateMember].isSet())
      return SK_PrivateMember;

    if (Decl->getAccess() == AS_protected &&
        NamingStyles[SK_ProtectedMember].isSet())
      return SK_ProtectedMember;

    if (Decl->getAccess() == AS_public && NamingStyles[SK_PublicMember].isSet())
      return SK_PublicMember;

    if (NamingStyles[SK_Member].isSet())
      return SK_Member;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<ParmVarDecl>(D)) {
    QualType Type = Decl->getType();

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprVariable].isSet())
      return SK_ConstexprVariable;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        NamingStyles[SK_ConstantParameter].isSet())
      return SK_ConstantParameter;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        NamingStyles[SK_Constant].isSet())
      return SK_Constant;

    if (Decl->isParameterPack() && NamingStyles[SK_ParameterPack].isSet())
      return SK_ParameterPack;

    if (NamingStyles[SK_Parameter].isSet())
      return SK_Parameter;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<VarDecl>(D)) {
    QualType Type = Decl->getType();

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprVariable].isSet())
      return SK_ConstexprVariable;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        Decl->isStaticDataMember() && NamingStyles[SK_ClassConstant].isSet())
      return SK_ClassConstant;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        Decl->isFileVarDecl() && NamingStyles[SK_GlobalConstant].isSet())
      return SK_GlobalConstant;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        Decl->isStaticLocal() && NamingStyles[SK_StaticConstant].isSet())
      return SK_StaticConstant;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        Decl->isLocalVarDecl() && NamingStyles[SK_LocalConstant].isSet())
      return SK_LocalConstant;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        Decl->isFunctionOrMethodVarDecl() &&
        NamingStyles[SK_LocalConstant].isSet())
      return SK_LocalConstant;

    if (!Type.isNull() && Type.isLocalConstQualified() &&
        NamingStyles[SK_Constant].isSet())
      return SK_Constant;

    if (Decl->isStaticDataMember() && NamingStyles[SK_ClassMember].isSet())
      return SK_ClassMember;

    if (Decl->isFileVarDecl() && NamingStyles[SK_GlobalVariable].isSet())
      return SK_GlobalVariable;

    if (Decl->isStaticLocal() && NamingStyles[SK_StaticVariable].isSet())
      return SK_StaticVariable;

    if (Decl->isLocalVarDecl() && NamingStyles[SK_LocalVariable].isSet())
      return SK_LocalVariable;

    if (Decl->isFunctionOrMethodVarDecl() &&
        NamingStyles[SK_LocalVariable].isSet())
      return SK_LocalVariable;

    if (NamingStyles[SK_Variable].isSet())
      return SK_Variable;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<CXXMethodDecl>(D)) {
    if (Decl->isMain() || !Decl->isUserProvided() ||
        Decl->isUsualDeallocationFunction() ||
        Decl->isCopyAssignmentOperator() || Decl->isMoveAssignmentOperator() ||
        Decl->size_overridden_methods() > 0)
      return SK_Invalid;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprMethod].isSet())
      return SK_ConstexprMethod;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprFunction].isSet())
      return SK_ConstexprFunction;

    if (Decl->isStatic() && NamingStyles[SK_ClassMethod].isSet())
      return SK_ClassMethod;

    if (Decl->isVirtual() && NamingStyles[SK_VirtualMethod].isSet())
      return SK_VirtualMethod;

    if (Decl->getAccess() == AS_private &&
        NamingStyles[SK_PrivateMethod].isSet())
      return SK_PrivateMethod;

    if (Decl->getAccess() == AS_protected &&
        NamingStyles[SK_ProtectedMethod].isSet())
      return SK_ProtectedMethod;

    if (Decl->getAccess() == AS_public && NamingStyles[SK_PublicMethod].isSet())
      return SK_PublicMethod;

    if (NamingStyles[SK_Method].isSet())
      return SK_Method;

    if (NamingStyles[SK_Function].isSet())
      return SK_Function;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<FunctionDecl>(D)) {
    if (Decl->isMain())
      return SK_Invalid;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprFunction].isSet())
      return SK_ConstexprFunction;

    if (Decl->isGlobal() && NamingStyles[SK_GlobalFunction].isSet())
      return SK_GlobalFunction;

    if (NamingStyles[SK_Function].isSet())
      return SK_Function;
  }

  if (isa<TemplateTypeParmDecl>(D)) {
    if (NamingStyles[SK_TypeTemplateParameter].isSet())
      return SK_TypeTemplateParameter;

    if (NamingStyles[SK_TemplateParameter].isSet())
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  if (isa<NonTypeTemplateParmDecl>(D)) {
    if (NamingStyles[SK_ValueTemplateParameter].isSet())
      return SK_ValueTemplateParameter;

    if (NamingStyles[SK_TemplateParameter].isSet())
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  if (isa<TemplateTemplateParmDecl>(D)) {
    if (NamingStyles[SK_TemplateTemplateParameter].isSet())
      return SK_TemplateTemplateParameter;

    if (NamingStyles[SK_TemplateParameter].isSet())
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  return SK_Invalid;
}

static void addUsage(IdentifierNamingCheck::NamingCheckFailureMap &Failures,
                     const IdentifierNamingCheck::NamingCheckId &Decl,
                     SourceRange Range) {
  // Do nothing if the provided range is invalid.
  if (Range.getBegin().isInvalid() || Range.getEnd().isInvalid())
    return;

  // Try to insert the identifier location in the Usages map, and bail out if it
  // is already in there
  auto &Failure = Failures[Decl];
  if (!Failure.RawUsageLocs.insert(Range.getBegin().getRawEncoding()).second)
    return;

  Failure.ShouldFix = Failure.ShouldFix && !Range.getBegin().isMacroID() &&
                      !Range.getEnd().isMacroID();
}

/// Convenience method when the usage to be added is a NamedDecl
static void addUsage(IdentifierNamingCheck::NamingCheckFailureMap &Failures,
                     const NamedDecl *Decl, SourceRange Range) {
  return addUsage(Failures, IdentifierNamingCheck::NamingCheckId(
                                Decl->getLocation(), Decl->getNameAsString()),
                  Range);
}

void IdentifierNamingCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl =
          Result.Nodes.getNodeAs<CXXConstructorDecl>("classRef")) {
    if (Decl->isImplicit())
      return;

    addUsage(NamingCheckFailures, Decl->getParent(),
             Decl->getNameInfo().getSourceRange());
    return;
  }

  if (const auto *Decl =
          Result.Nodes.getNodeAs<CXXDestructorDecl>("classRef")) {
    if (Decl->isImplicit())
      return;

    SourceRange Range = Decl->getNameInfo().getSourceRange();
    if (Range.getBegin().isInvalid())
      return;
    // The first token that will be found is the ~ (or the equivalent trigraph),
    // we want instead to replace the next token, that will be the identifier.
    Range.setBegin(CharSourceRange::getTokenRange(Range).getEnd());

    addUsage(NamingCheckFailures, Decl->getParent(), Range);
    return;
  }

  if (const auto *Loc = Result.Nodes.getNodeAs<TypeLoc>("typeLoc")) {
    NamedDecl *Decl = nullptr;
    if (const auto &Ref = Loc->getAs<TagTypeLoc>()) {
      Decl = Ref.getDecl();
    } else if (const auto &Ref = Loc->getAs<InjectedClassNameTypeLoc>()) {
      Decl = Ref.getDecl();
    } else if (const auto &Ref = Loc->getAs<UnresolvedUsingTypeLoc>()) {
      Decl = Ref.getDecl();
    } else if (const auto &Ref = Loc->getAs<TemplateTypeParmTypeLoc>()) {
      Decl = Ref.getDecl();
    }

    if (Decl) {
      addUsage(NamingCheckFailures, Decl, Loc->getSourceRange());
      return;
    }

    if (const auto &Ref = Loc->getAs<TemplateSpecializationTypeLoc>()) {
      const auto *Decl =
          Ref.getTypePtr()->getTemplateName().getAsTemplateDecl();

      SourceRange Range(Ref.getTemplateNameLoc(), Ref.getTemplateNameLoc());
      if (const auto *ClassDecl = dyn_cast<TemplateDecl>(Decl)) {
        if (const auto *TemplDecl = ClassDecl->getTemplatedDecl())
          addUsage(NamingCheckFailures, TemplDecl, Range);
        return;
      }
    }

    if (const auto &Ref =
            Loc->getAs<DependentTemplateSpecializationTypeLoc>()) {
      if (const auto *Decl = Ref.getTypePtr()->getAsTagDecl())
        addUsage(NamingCheckFailures, Decl, Loc->getSourceRange());
      return;
    }
  }

  if (const auto *Loc =
          Result.Nodes.getNodeAs<NestedNameSpecifierLoc>("nestedNameLoc")) {
    if (NestedNameSpecifier *Spec = Loc->getNestedNameSpecifier()) {
      if (NamespaceDecl *Decl = Spec->getAsNamespace()) {
        addUsage(NamingCheckFailures, Decl, Loc->getLocalSourceRange());
        return;
      }
    }
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<UsingDecl>("using")) {
    for (const auto &Shadow : Decl->shadows()) {
      addUsage(NamingCheckFailures, Shadow->getTargetDecl(),
               Decl->getNameInfo().getSourceRange());
    }
    return;
  }

  if (const auto *DeclRef = Result.Nodes.getNodeAs<DeclRefExpr>("declRef")) {
    SourceRange Range = DeclRef->getNameInfo().getSourceRange();
    addUsage(NamingCheckFailures, DeclRef->getDecl(), Range);
    return;
  }

  if (const auto *Decl = Result.Nodes.getNodeAs<NamedDecl>("decl")) {
    if (!Decl->getIdentifier() || Decl->getName().empty() || Decl->isImplicit())
      return;

    // Fix type aliases in value declarations
    if (const auto *Value = Result.Nodes.getNodeAs<ValueDecl>("decl")) {
      if (const auto *Typedef =
              Value->getType().getTypePtr()->getAs<TypedefType>()) {
        addUsage(NamingCheckFailures, Typedef->getDecl(),
                 Value->getSourceRange());
      }
    }

    // Fix type aliases in function declarations
    if (const auto *Value = Result.Nodes.getNodeAs<FunctionDecl>("decl")) {
      if (const auto *Typedef =
              Value->getReturnType().getTypePtr()->getAs<TypedefType>()) {
        addUsage(NamingCheckFailures, Typedef->getDecl(),
                 Value->getSourceRange());
      }
      for (unsigned i = 0; i < Value->getNumParams(); ++i) {
        if (const auto *Typedef = Value->parameters()[i]
                                      ->getType()
                                      .getTypePtr()
                                      ->getAs<TypedefType>()) {
          addUsage(NamingCheckFailures, Typedef->getDecl(),
                   Value->getSourceRange());
        }
      }
    }

    // Ignore ClassTemplateSpecializationDecl which are creating duplicate
    // replacements with CXXRecordDecl
    if (isa<ClassTemplateSpecializationDecl>(Decl))
      return;

    StyleKind SK = findStyleKind(Decl, NamingStyles);
    if (SK == SK_Invalid)
      return;

    NamingStyle Style = NamingStyles[SK];
    StringRef Name = Decl->getName();
    if (matchesStyle(Name, Style))
      return;

    std::string KindName = fixupWithCase(StyleNames[SK], CT_LowerCase);
    std::replace(KindName.begin(), KindName.end(), '_', ' ');

    std::string Fixup = fixupWithStyle(Name, Style);
    if (StringRef(Fixup).equals(Name)) {
      if (!IgnoreFailedSplit) {
        DEBUG(llvm::dbgs()
              << Decl->getLocStart().printToString(*Result.SourceManager)
              << llvm::format(": unable to split words for %s '%s'\n",
                              KindName.c_str(), Name));
      }
    } else {
      NamingCheckFailure &Failure = NamingCheckFailures[NamingCheckId(
          Decl->getLocation(), Decl->getNameAsString())];
      SourceRange Range =
          DeclarationNameInfo(Decl->getDeclName(), Decl->getLocation())
              .getSourceRange();

      Failure.Fixup = std::move(Fixup);
      Failure.KindName = std::move(KindName);
      addUsage(NamingCheckFailures, Decl, Range);
    }
  }
}

void IdentifierNamingCheck::checkMacro(SourceManager &SourceMgr,
                                       const Token &MacroNameTok,
                                       const MacroInfo *MI) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingStyle Style = NamingStyles[SK_MacroDefinition];
  if (matchesStyle(Name, Style))
    return;

  std::string KindName =
      fixupWithCase(StyleNames[SK_MacroDefinition], CT_LowerCase);
  std::replace(KindName.begin(), KindName.end(), '_', ' ');

  std::string Fixup = fixupWithStyle(Name, Style);
  if (StringRef(Fixup).equals(Name)) {
    if (!IgnoreFailedSplit) {
      DEBUG(
          llvm::dbgs() << MacroNameTok.getLocation().printToString(SourceMgr)
                       << llvm::format(": unable to split words for %s '%s'\n",
                                       KindName.c_str(), Name));
    }
  } else {
    NamingCheckId ID(MI->getDefinitionLoc(), Name);
    NamingCheckFailure &Failure = NamingCheckFailures[ID];
    SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());

    Failure.Fixup = std::move(Fixup);
    Failure.KindName = std::move(KindName);
    addUsage(NamingCheckFailures, ID, Range);
  }
}

void IdentifierNamingCheck::expandMacro(const Token &MacroNameTok,
                                        const MacroInfo *MI) {
  StringRef Name = MacroNameTok.getIdentifierInfo()->getName();
  NamingCheckId ID(MI->getDefinitionLoc(), Name);

  auto Failure = NamingCheckFailures.find(ID);
  if (Failure == NamingCheckFailures.end())
    return;

  SourceRange Range(MacroNameTok.getLocation(), MacroNameTok.getEndLoc());
  addUsage(NamingCheckFailures, ID, Range);
}

void IdentifierNamingCheck::onEndOfTranslationUnit() {
  for (const auto &Pair : NamingCheckFailures) {
    const NamingCheckId &Decl = Pair.first;
    const NamingCheckFailure &Failure = Pair.second;

    if (Failure.KindName.empty())
      continue;

    if (Failure.ShouldFix) {
      auto Diag = diag(Decl.first, "invalid case style for %0 '%1'")
                  << Failure.KindName << Decl.second;

      for (const auto &Loc : Failure.RawUsageLocs) {
        // We assume that the identifier name is made of one token only. This is
        // always the case as we ignore usages in macros that could build
        // identifier names by combining multiple tokens.
        //
        // For destructors, we alread take care of it by remembering the
        // location of the start of the identifier and not the start of the
        // tilde.
        //
        // Other multi-token identifiers, such as operators are not checked at
        // all.
        Diag << FixItHint::CreateReplacement(
            SourceRange(SourceLocation::getFromRawEncoding(Loc)),
            Failure.Fixup);
      }
    }
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
