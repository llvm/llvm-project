//===--- IdentifierNamingCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdentifierNamingCheck.h"

#include "../GlobList.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "clang-tidy"

// FixItHint

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

llvm::ArrayRef<
    std::pair<readability::IdentifierNamingCheck::CaseType, StringRef>>
OptionEnumMapping<
    readability::IdentifierNamingCheck::CaseType>::getEnumMapping() {
  static constexpr std::pair<readability::IdentifierNamingCheck::CaseType,
                             StringRef>
      Mapping[] = {
          {readability::IdentifierNamingCheck::CT_AnyCase, "aNy_CasE"},
          {readability::IdentifierNamingCheck::CT_LowerCase, "lower_case"},
          {readability::IdentifierNamingCheck::CT_UpperCase, "UPPER_CASE"},
          {readability::IdentifierNamingCheck::CT_CamelBack, "camelBack"},
          {readability::IdentifierNamingCheck::CT_CamelCase, "CamelCase"},
          {readability::IdentifierNamingCheck::CT_CamelSnakeCase,
           "Camel_Snake_Case"},
          {readability::IdentifierNamingCheck::CT_CamelSnakeBack,
           "camel_Snake_Back"}};
  return llvm::makeArrayRef(Mapping);
}

namespace readability {

// clang-format off
#define NAMING_KEYS(m) \
    m(Namespace) \
    m(InlineNamespace) \
    m(EnumConstant) \
    m(ScopedEnumConstant) \
    m(ConstexprVariable) \
    m(ConstantMember) \
    m(PrivateMember) \
    m(ProtectedMember) \
    m(PublicMember) \
    m(Member) \
    m(ClassConstant) \
    m(ClassMember) \
    m(GlobalConstant) \
    m(GlobalConstantPointer) \
    m(GlobalPointer) \
    m(GlobalVariable) \
    m(LocalConstant) \
    m(LocalConstantPointer) \
    m(LocalPointer) \
    m(LocalVariable) \
    m(StaticConstant) \
    m(StaticVariable) \
    m(Constant) \
    m(Variable) \
    m(ConstantParameter) \
    m(ParameterPack) \
    m(Parameter) \
    m(PointerParameter) \
    m(ConstantPointerParameter) \
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
    m(ObjcIvar) \

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

IdentifierNamingCheck::NamingStyle::NamingStyle(
    llvm::Optional<IdentifierNamingCheck::CaseType> Case,
    const std::string &Prefix, const std::string &Suffix,
    const std::string &IgnoredRegexpStr)
    : Case(Case), Prefix(Prefix), Suffix(Suffix),
      IgnoredRegexpStr(IgnoredRegexpStr) {
  if (!IgnoredRegexpStr.empty()) {
    IgnoredRegexp =
        llvm::Regex(llvm::SmallString<128>({"^", IgnoredRegexpStr, "$"}));
    if (!IgnoredRegexp.isValid())
      llvm::errs() << "Invalid IgnoredRegexp regular expression: "
                   << IgnoredRegexpStr;
  }
}

static IdentifierNamingCheck::FileStyle
getFileStyleFromOptions(const ClangTidyCheck::OptionsView &Options) {
  SmallVector<llvm::Optional<IdentifierNamingCheck::NamingStyle>, 0> Styles;
  Styles.resize(SK_Count);
  SmallString<64> StyleString;
  for (unsigned I = 0; I < SK_Count; ++I) {
    StyleString = StyleNames[I];
    size_t StyleSize = StyleString.size();
    StyleString.append("IgnoredRegexp");
    std::string IgnoredRegexpStr = Options.get(StyleString, "");
    StyleString.resize(StyleSize);
    StyleString.append("Prefix");
    std::string Prefix(Options.get(StyleString, ""));
    // Fast replacement of [Pre]fix -> [Suf]fix.
    memcpy(&StyleString[StyleSize], "Suf", 3);
    std::string Postfix(Options.get(StyleString, ""));
    memcpy(&StyleString[StyleSize], "Case", 4);
    StyleString.pop_back();
    StyleString.pop_back();
    auto CaseOptional =
        Options.get<IdentifierNamingCheck::CaseType>(StyleString);

    if (CaseOptional || !Prefix.empty() || !Postfix.empty() ||
        !IgnoredRegexpStr.empty())
      Styles[I].emplace(std::move(CaseOptional), std::move(Prefix),
                        std::move(Postfix), std::move(IgnoredRegexpStr));
  }
  bool IgnoreMainLike = Options.get("IgnoreMainLikeFunctions", false);
  return {std::move(Styles), IgnoreMainLike};
}

IdentifierNamingCheck::IdentifierNamingCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : RenamerClangTidyCheck(Name, Context), Context(Context), CheckName(Name),
      GetConfigPerFile(Options.get("GetConfigPerFile", true)),
      IgnoreFailedSplit(Options.get("IgnoreFailedSplit", false)) {

  auto IterAndInserted = NamingStylesCache.try_emplace(
      llvm::sys::path::parent_path(Context->getCurrentFile()),
      getFileStyleFromOptions(Options));
  assert(IterAndInserted.second && "Couldn't insert Style");
  // Holding a reference to the data in the vector is safe as it should never
  // move.
  MainFileStyle = &IterAndInserted.first->getValue();
}

IdentifierNamingCheck::~IdentifierNamingCheck() = default;

void IdentifierNamingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  RenamerClangTidyCheck::storeOptions(Opts);
  SmallString<64> StyleString;
  ArrayRef<llvm::Optional<NamingStyle>> Styles = MainFileStyle->getStyles();
  for (size_t I = 0; I < SK_Count; ++I) {
    if (!Styles[I])
      continue;
    StyleString = StyleNames[I];
    size_t StyleSize = StyleString.size();
    StyleString.append("IgnoredRegexp");
    Options.store(Opts, StyleString, Styles[I]->IgnoredRegexpStr);
    StyleString.resize(StyleSize);
    StyleString.append("Prefix");
    Options.store(Opts, StyleString, Styles[I]->Prefix);
    // Fast replacement of [Pre]fix -> [Suf]fix.
    memcpy(&StyleString[StyleSize], "Suf", 3);
    Options.store(Opts, StyleString, Styles[I]->Suffix);
    if (Styles[I]->Case) {
      memcpy(&StyleString[StyleSize], "Case", 4);
      StyleString.pop_back();
      StyleString.pop_back();
      Options.store(Opts, StyleString, *Styles[I]->Case);
    }
  }
  Options.store(Opts, "GetConfigPerFile", GetConfigPerFile);
  Options.store(Opts, "IgnoreFailedSplit", IgnoreFailedSplit);
  Options.store(Opts, "IgnoreMainLikeFunctions",
                MainFileStyle->isIgnoringMainLikeFunction());
}

static bool matchesStyle(StringRef Name,
                         const IdentifierNamingCheck::NamingStyle &Style) {
  static llvm::Regex Matchers[] = {
      llvm::Regex("^.*$"),
      llvm::Regex("^[a-z][a-z0-9_]*$"),
      llvm::Regex("^[a-z][a-zA-Z0-9]*$"),
      llvm::Regex("^[A-Z][A-Z0-9_]*$"),
      llvm::Regex("^[A-Z][a-zA-Z0-9]*$"),
      llvm::Regex("^[A-Z]([a-z0-9]*(_[A-Z])?)*"),
      llvm::Regex("^[a-z]([a-z0-9]*(_[A-Z])?)*"),
  };

  if (!Name.consume_front(Style.Prefix))
    return false;
  if (!Name.consume_back(Style.Suffix))
    return false;

  // Ensure the name doesn't have any extra underscores beyond those specified
  // in the prefix and suffix.
  if (Name.startswith("_") || Name.endswith("_"))
    return false;

  if (Style.Case && !Matchers[static_cast<size_t>(*Style.Case)].match(Name))
    return false;

  return true;
}

static std::string fixupWithCase(StringRef Name,
                                 IdentifierNamingCheck::CaseType Case) {
  static llvm::Regex Splitter(
      "([a-z0-9A-Z]*)(_+)|([A-Z]?[a-z0-9]+)([A-Z]|$)|([A-Z]+)([A-Z]|$)");

  SmallVector<StringRef, 8> Substrs;
  Name.split(Substrs, "_", -1, false);

  SmallVector<StringRef, 8> Words;
  SmallVector<StringRef, 8> Groups;
  for (auto Substr : Substrs) {
    while (!Substr.empty()) {
      Groups.clear();
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
    return Name.str();

  SmallString<128> Fixup;
  switch (Case) {
  case IdentifierNamingCheck::CT_AnyCase:
    return Name.str();
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
      Fixup += toupper(Word.front());
      Fixup += Word.substr(1).lower();
    }
    break;

  case IdentifierNamingCheck::CT_CamelBack:
    for (auto const &Word : Words) {
      if (&Word == &Words.front()) {
        Fixup += Word.lower();
      } else {
        Fixup += toupper(Word.front());
        Fixup += Word.substr(1).lower();
      }
    }
    break;

  case IdentifierNamingCheck::CT_CamelSnakeCase:
    for (auto const &Word : Words) {
      if (&Word != &Words.front())
        Fixup += "_";
      Fixup += toupper(Word.front());
      Fixup += Word.substr(1).lower();
    }
    break;

  case IdentifierNamingCheck::CT_CamelSnakeBack:
    for (auto const &Word : Words) {
      if (&Word != &Words.front()) {
        Fixup += "_";
        Fixup += toupper(Word.front());
      } else {
        Fixup += tolower(Word.front());
      }
      Fixup += Word.substr(1).lower();
    }
    break;
  }

  return Fixup.str().str();
}

static bool isParamInMainLikeFunction(const ParmVarDecl &ParmDecl,
                                      bool IncludeMainLike) {
  const auto *FDecl =
      dyn_cast_or_null<FunctionDecl>(ParmDecl.getParentFunctionOrMethod());
  if (!FDecl)
    return false;
  if (FDecl->isMain())
    return true;
  if (!IncludeMainLike)
    return false;
  if (FDecl->getAccess() != AS_public && FDecl->getAccess() != AS_none)
    return false;
  // If the function doesn't have a name that's an identifier, can occur if the
  // function is an operator overload, bail out early.
  if (!FDecl->getDeclName().isIdentifier())
    return false;
  enum MainType { None, Main, WMain };
  auto IsCharPtrPtr = [](QualType QType) -> MainType {
    if (QType.isNull())
      return None;
    if (QType = QType->getPointeeType(), QType.isNull())
      return None;
    if (QType = QType->getPointeeType(), QType.isNull())
      return None;
    if (QType->isCharType())
      return Main;
    if (QType->isWideCharType())
      return WMain;
    return None;
  };
  auto IsIntType = [](QualType QType) {
    if (QType.isNull())
      return false;
    if (const auto *Builtin =
            dyn_cast<BuiltinType>(QType->getUnqualifiedDesugaredType())) {
      return Builtin->getKind() == BuiltinType::Int;
    }
    return false;
  };
  if (!IsIntType(FDecl->getReturnType()))
    return false;
  if (FDecl->getNumParams() < 2 || FDecl->getNumParams() > 3)
    return false;
  if (!IsIntType(FDecl->parameters()[0]->getType()))
    return false;
  MainType Type = IsCharPtrPtr(FDecl->parameters()[1]->getType());
  if (Type == None)
    return false;
  if (FDecl->getNumParams() == 3 &&
      IsCharPtrPtr(FDecl->parameters()[2]->getType()) != Type)
    return false;

  if (Type == Main) {
    static llvm::Regex Matcher(
        "(^[Mm]ain([_A-Z]|$))|([a-z0-9_]Main([_A-Z]|$))|(_main(_|$))");
    assert(Matcher.isValid() && "Invalid Matcher for main like functions.");
    return Matcher.match(FDecl->getName());
  }
  static llvm::Regex Matcher("(^((W[Mm])|(wm))ain([_A-Z]|$))|([a-z0-9_]W[Mm]"
                             "ain([_A-Z]|$))|(_wmain(_|$))");
  assert(Matcher.isValid() && "Invalid Matcher for wmain like functions.");
  return Matcher.match(FDecl->getName());
}

static std::string
fixupWithStyle(StringRef Name,
               const IdentifierNamingCheck::NamingStyle &Style) {
  const std::string Fixed = fixupWithCase(
      Name, Style.Case.getValueOr(IdentifierNamingCheck::CaseType::CT_AnyCase));
  StringRef Mid = StringRef(Fixed).trim("_");
  if (Mid.empty())
    Mid = "_";
  return (Style.Prefix + Mid + Style.Suffix).str();
}

static StyleKind findStyleKind(
    const NamedDecl *D,
    ArrayRef<llvm::Optional<IdentifierNamingCheck::NamingStyle>> NamingStyles,
    bool IgnoreMainLikeFunctions) {
  assert(D && D->getIdentifier() && !D->getName().empty() && !D->isImplicit() &&
         "Decl must be an explicit identifier with a name.");

  if (isa<ObjCIvarDecl>(D) && NamingStyles[SK_ObjcIvar])
    return SK_ObjcIvar;
  
  if (isa<TypedefDecl>(D) && NamingStyles[SK_Typedef])
    return SK_Typedef;

  if (isa<TypeAliasDecl>(D) && NamingStyles[SK_TypeAlias])
    return SK_TypeAlias;

  if (const auto *Decl = dyn_cast<NamespaceDecl>(D)) {
    if (Decl->isAnonymousNamespace())
      return SK_Invalid;

    if (Decl->isInline() && NamingStyles[SK_InlineNamespace])
      return SK_InlineNamespace;

    if (NamingStyles[SK_Namespace])
      return SK_Namespace;
  }

  if (isa<EnumDecl>(D) && NamingStyles[SK_Enum])
    return SK_Enum;

  if (const auto *EnumConst = dyn_cast<EnumConstantDecl>(D)) {
    if (cast<EnumDecl>(EnumConst->getDeclContext())->isScoped() &&
        NamingStyles[SK_ScopedEnumConstant])
      return SK_ScopedEnumConstant;

    if (NamingStyles[SK_EnumConstant])
      return SK_EnumConstant;

    if (NamingStyles[SK_Constant])
      return SK_Constant;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<CXXRecordDecl>(D)) {
    if (Decl->isAnonymousStructOrUnion())
      return SK_Invalid;

    if (!Decl->getCanonicalDecl()->isThisDeclarationADefinition())
      return SK_Invalid;

    if (Decl->hasDefinition() && Decl->isAbstract() &&
        NamingStyles[SK_AbstractClass])
      return SK_AbstractClass;

    if (Decl->isStruct() && NamingStyles[SK_Struct])
      return SK_Struct;

    if (Decl->isStruct() && NamingStyles[SK_Class])
      return SK_Class;

    if (Decl->isClass() && NamingStyles[SK_Class])
      return SK_Class;

    if (Decl->isClass() && NamingStyles[SK_Struct])
      return SK_Struct;

    if (Decl->isUnion() && NamingStyles[SK_Union])
      return SK_Union;

    if (Decl->isEnum() && NamingStyles[SK_Enum])
      return SK_Enum;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<FieldDecl>(D)) {
    QualType Type = Decl->getType();

    if (!Type.isNull() && Type.isConstQualified()) {
      if (NamingStyles[SK_ConstantMember])
        return SK_ConstantMember;

      if (NamingStyles[SK_Constant])
        return SK_Constant;
    }

    if (Decl->getAccess() == AS_private && NamingStyles[SK_PrivateMember])
      return SK_PrivateMember;

    if (Decl->getAccess() == AS_protected && NamingStyles[SK_ProtectedMember])
      return SK_ProtectedMember;

    if (Decl->getAccess() == AS_public && NamingStyles[SK_PublicMember])
      return SK_PublicMember;

    if (NamingStyles[SK_Member])
      return SK_Member;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<ParmVarDecl>(D)) {
    if (isParamInMainLikeFunction(*Decl, IgnoreMainLikeFunctions))
      return SK_Invalid;
    QualType Type = Decl->getType();

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprVariable])
      return SK_ConstexprVariable;

    if (!Type.isNull() && Type.isConstQualified()) {
      if (Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_ConstantPointerParameter])
        return SK_ConstantPointerParameter;

      if (NamingStyles[SK_ConstantParameter])
        return SK_ConstantParameter;

      if (NamingStyles[SK_Constant])
        return SK_Constant;
    }

    if (Decl->isParameterPack() && NamingStyles[SK_ParameterPack])
      return SK_ParameterPack;

    if (!Type.isNull() && Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_PointerParameter])
        return SK_PointerParameter;

    if (NamingStyles[SK_Parameter])
      return SK_Parameter;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<VarDecl>(D)) {
    QualType Type = Decl->getType();

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprVariable])
      return SK_ConstexprVariable;

    if (!Type.isNull() && Type.isConstQualified()) {
      if (Decl->isStaticDataMember() && NamingStyles[SK_ClassConstant])
        return SK_ClassConstant;

      if (Decl->isFileVarDecl() && Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_GlobalConstantPointer])
        return SK_GlobalConstantPointer;

      if (Decl->isFileVarDecl() && NamingStyles[SK_GlobalConstant])
        return SK_GlobalConstant;

      if (Decl->isStaticLocal() && NamingStyles[SK_StaticConstant])
        return SK_StaticConstant;

      if (Decl->isLocalVarDecl() && Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_LocalConstantPointer])
        return SK_LocalConstantPointer;

      if (Decl->isLocalVarDecl() && NamingStyles[SK_LocalConstant])
        return SK_LocalConstant;

      if (Decl->isFunctionOrMethodVarDecl() && NamingStyles[SK_LocalConstant])
        return SK_LocalConstant;

      if (NamingStyles[SK_Constant])
        return SK_Constant;
    }

    if (Decl->isStaticDataMember() && NamingStyles[SK_ClassMember])
      return SK_ClassMember;

    if (Decl->isFileVarDecl() && Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_GlobalPointer])
      return SK_GlobalPointer;

    if (Decl->isFileVarDecl() && NamingStyles[SK_GlobalVariable])
      return SK_GlobalVariable;

    if (Decl->isStaticLocal() && NamingStyles[SK_StaticVariable])
      return SK_StaticVariable;
 
    if (Decl->isLocalVarDecl() && Type.getTypePtr()->isAnyPointerType() && NamingStyles[SK_LocalPointer])
      return SK_LocalPointer;

    if (Decl->isLocalVarDecl() && NamingStyles[SK_LocalVariable])
      return SK_LocalVariable;

    if (Decl->isFunctionOrMethodVarDecl() && NamingStyles[SK_LocalVariable])
      return SK_LocalVariable;

    if (NamingStyles[SK_Variable])
      return SK_Variable;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<CXXMethodDecl>(D)) {
    if (Decl->isMain() || !Decl->isUserProvided() ||
        Decl->size_overridden_methods() > 0)
      return SK_Invalid;

    // If this method has the same name as any base method, this is likely
    // necessary even if it's not an override. e.g. CRTP.
    for (const CXXBaseSpecifier &Base : Decl->getParent()->bases())
      if (const auto *RD = Base.getType()->getAsCXXRecordDecl())
        if (RD->hasMemberName(Decl->getDeclName()))
          return SK_Invalid;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprMethod])
      return SK_ConstexprMethod;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprFunction])
      return SK_ConstexprFunction;

    if (Decl->isStatic() && NamingStyles[SK_ClassMethod])
      return SK_ClassMethod;

    if (Decl->isVirtual() && NamingStyles[SK_VirtualMethod])
      return SK_VirtualMethod;

    if (Decl->getAccess() == AS_private && NamingStyles[SK_PrivateMethod])
      return SK_PrivateMethod;

    if (Decl->getAccess() == AS_protected && NamingStyles[SK_ProtectedMethod])
      return SK_ProtectedMethod;

    if (Decl->getAccess() == AS_public && NamingStyles[SK_PublicMethod])
      return SK_PublicMethod;

    if (NamingStyles[SK_Method])
      return SK_Method;

    if (NamingStyles[SK_Function])
      return SK_Function;

    return SK_Invalid;
  }

  if (const auto *Decl = dyn_cast<FunctionDecl>(D)) {
    if (Decl->isMain())
      return SK_Invalid;

    if (Decl->isConstexpr() && NamingStyles[SK_ConstexprFunction])
      return SK_ConstexprFunction;

    if (Decl->isGlobal() && NamingStyles[SK_GlobalFunction])
      return SK_GlobalFunction;

    if (NamingStyles[SK_Function])
      return SK_Function;
  }

  if (isa<TemplateTypeParmDecl>(D)) {
    if (NamingStyles[SK_TypeTemplateParameter])
      return SK_TypeTemplateParameter;

    if (NamingStyles[SK_TemplateParameter])
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  if (isa<NonTypeTemplateParmDecl>(D)) {
    if (NamingStyles[SK_ValueTemplateParameter])
      return SK_ValueTemplateParameter;

    if (NamingStyles[SK_TemplateParameter])
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  if (isa<TemplateTemplateParmDecl>(D)) {
    if (NamingStyles[SK_TemplateTemplateParameter])
      return SK_TemplateTemplateParameter;

    if (NamingStyles[SK_TemplateParameter])
      return SK_TemplateParameter;

    return SK_Invalid;
  }

  return SK_Invalid;
}

static llvm::Optional<RenamerClangTidyCheck::FailureInfo> getFailureInfo(
    StringRef Name, SourceLocation Location,
    ArrayRef<llvm::Optional<IdentifierNamingCheck::NamingStyle>> NamingStyles,
    StyleKind SK, const SourceManager &SM, bool IgnoreFailedSplit) {
  if (SK == SK_Invalid || !NamingStyles[SK])
    return None;

  const IdentifierNamingCheck::NamingStyle &Style = *NamingStyles[SK];
  if (Style.IgnoredRegexp.isValid() && Style.IgnoredRegexp.match(Name))
    return None;

  if (matchesStyle(Name, Style))
    return None;

  std::string KindName =
      fixupWithCase(StyleNames[SK], IdentifierNamingCheck::CT_LowerCase);
  std::replace(KindName.begin(), KindName.end(), '_', ' ');

  std::string Fixup = fixupWithStyle(Name, Style);
  if (StringRef(Fixup).equals(Name)) {
    if (!IgnoreFailedSplit) {
      LLVM_DEBUG(Location.print(llvm::dbgs(), SM);
                 llvm::dbgs()
                 << llvm::formatv(": unable to split words for {0} '{1}'\n",
                                  KindName, Name));
    }
    return None;
  }
  return RenamerClangTidyCheck::FailureInfo{std::move(KindName),
                                            std::move(Fixup)};
}

llvm::Optional<RenamerClangTidyCheck::FailureInfo>
IdentifierNamingCheck::GetDeclFailureInfo(const NamedDecl *Decl,
                                          const SourceManager &SM) const {
  SourceLocation Loc = Decl->getLocation();
  const FileStyle &FileStyle = getStyleForFile(SM.getFilename(Loc));
  if (!FileStyle.isActive())
    return llvm::None;

  return getFailureInfo(Decl->getName(), Loc, FileStyle.getStyles(),
                        findStyleKind(Decl, FileStyle.getStyles(),
                                      FileStyle.isIgnoringMainLikeFunction()),
                        SM, IgnoreFailedSplit);
}

llvm::Optional<RenamerClangTidyCheck::FailureInfo>
IdentifierNamingCheck::GetMacroFailureInfo(const Token &MacroNameTok,
                                           const SourceManager &SM) const {
  SourceLocation Loc = MacroNameTok.getLocation();
  const FileStyle &Style = getStyleForFile(SM.getFilename(Loc));
  if (!Style.isActive())
    return llvm::None;

  return getFailureInfo(MacroNameTok.getIdentifierInfo()->getName(), Loc,
                        Style.getStyles(), SK_MacroDefinition, SM,
                        IgnoreFailedSplit);
}

RenamerClangTidyCheck::DiagInfo
IdentifierNamingCheck::GetDiagInfo(const NamingCheckId &ID,
                                   const NamingCheckFailure &Failure) const {
  return DiagInfo{"invalid case style for %0 '%1'",
                  [&](DiagnosticBuilder &Diag) {
                    Diag << Failure.Info.KindName << ID.second;
                  }};
}

const IdentifierNamingCheck::FileStyle &
IdentifierNamingCheck::getStyleForFile(StringRef FileName) const {
  if (!GetConfigPerFile)
    return *MainFileStyle;
  StringRef Parent = llvm::sys::path::parent_path(FileName);
  auto Iter = NamingStylesCache.find(Parent);
  if (Iter != NamingStylesCache.end())
    return Iter->getValue();

  ClangTidyOptions Options = Context->getOptionsForFile(FileName);
  if (Options.Checks && GlobList(*Options.Checks).contains(CheckName)) {
    auto It = NamingStylesCache.try_emplace(
        Parent,
        getFileStyleFromOptions({CheckName, Options.CheckOptions, Context}));
    assert(It.second);
    return It.first->getValue();
  }
  // Default construction gives an empty style.
  auto It = NamingStylesCache.try_emplace(Parent);
  assert(It.second);
  return It.first->getValue();
}

} // namespace readability
} // namespace tidy
} // namespace clang
