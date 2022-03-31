//===--- ObjCMemberwiseInitializer.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "SourceCode.h"
#include "refactor/InsertionPoint.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {

static std::string capitalize(std::string Message) {
  if (!Message.empty())
    Message[0] = llvm::toUpper(Message[0]);
  return Message;
}

static std::string getTypeStr(const QualType &OrigT, const Decl &D,
                              unsigned PropertyAttributes) {
  QualType T = OrigT;
  PrintingPolicy Policy(D.getASTContext().getLangOpts());
  Policy.SuppressStrongLifetime = true;
  std::string Prefix = "";
  // If the nullability is specified via a property attribute, use the shorter
  // `nullable` form for the method parameter.
  if (PropertyAttributes & ObjCPropertyAttribute::kind_nullability) {
    if (auto Kind = AttributedType::stripOuterNullability(T)) {
      switch (Kind.getValue()) {
      case NullabilityKind::Nullable:
        Prefix = "nullable ";
        break;
      case NullabilityKind::NonNull:
        Prefix = "nonnull ";
        break;
      case NullabilityKind::Unspecified:
        Prefix = "null_unspecified ";
        break;
      case NullabilityKind::NullableResult:
        T = OrigT;
        break;
      }
    }
  }
  return Prefix + T.getAsString(Policy);
}

struct MethodParameter {
  // Parameter name.
  llvm::StringRef Name;

  // Type of the parameter.
  std::string Type;

  // Assignment target (LHS).
  std::string Assignee;

  MethodParameter(const ObjCIvarDecl &ID) {
    // Convention maps `@property int foo` to ivar `int _foo`, so drop the
    // leading `_` if there is one.
    Name = ID.getName();
    Name.consume_front("_");
    Type = getTypeStr(ID.getType(), ID, ObjCPropertyAttribute::kind_noattr);
    Assignee = ID.getName().str();
  }
  MethodParameter(const ObjCPropertyDecl &PD) {
    Name = PD.getName();
    Type = getTypeStr(PD.getType(), PD, PD.getPropertyAttributes());
    if (const auto *ID = PD.getPropertyIvarDecl())
      Assignee = ID->getName().str();
    else // Could be a dynamic property or a property in a header.
      Assignee = ("self." + Name).str();
  }
  static llvm::Optional<MethodParameter> parameterFor(const Decl &D) {
    if (const auto *ID = dyn_cast<ObjCIvarDecl>(&D))
      return MethodParameter(*ID);
    if (const auto *PD = dyn_cast<ObjCPropertyDecl>(&D))
      if (PD->isInstanceProperty())
        return MethodParameter(*PD);
    return llvm::None;
  }
};

static SmallVector<MethodParameter, 8>
getAllParams(const ObjCInterfaceDecl *ID) {
  SmallVector<MethodParameter, 8> Params;
  // Currently we only generate based on the ivars and properties declared
  // in the interface. We could consider expanding this to include visible
  // categories + class extensions in the future (see
  // all_declared_ivar_begin).
  llvm::DenseSet<llvm::StringRef> Names;
  for (const auto *Ivar : ID->ivars()) {
    MethodParameter P(*Ivar);
    if (Names.insert(P.Name).second)
      Params.push_back(P);
  }
  for (const auto *Prop : ID->properties()) {
    MethodParameter P(*Prop);
    if (Names.insert(P.Name).second)
      Params.push_back(P);
  }
  return Params;
}

static std::string
initializerForParams(const SmallVector<MethodParameter, 8> &Params,
                     bool GenerateImpl) {
  std::string Code;
  llvm::raw_string_ostream Stream(Code);

  if (Params.empty()) {
    if (GenerateImpl) {
      Stream <<
          R"cpp(- (instancetype)init {
  self = [super init];
  if (self) {

  }
  return self;
})cpp";
    } else {
      Stream << "- (instancetype)init;";
    }
  } else {
    const auto &First = Params.front();
    Stream << llvm::formatv("- (instancetype)initWith{0}:({1}){2}",
                            capitalize(First.Name.trim().str()), First.Type,
                            First.Name);
    for (auto It = Params.begin() + 1; It != Params.end(); ++It)
      Stream << llvm::formatv(" {0}:({1}){0}", It->Name, It->Type);

    if (GenerateImpl) {
      Stream <<
          R"cpp( {
  self = [super init];
  if (self) {)cpp";
      for (const auto &Param : Params)
        Stream << llvm::formatv("\n    {0} = {1};", Param.Assignee, Param.Name);
      Stream <<
          R"cpp(
  }
  return self;
})cpp";
    } else {
      Stream << ";";
    }
  }
  Stream << "\n\n";
  return Code;
}

/// Generate an initializer for an Objective-C class based on selected
/// properties and instance variables.
class ObjCMemberwiseInitializer : public Tweak {
public:
  const char *id() const override final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

  bool prepare(const Selection &Inputs) override;
  Expected<Tweak::Effect> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  SmallVector<MethodParameter, 8>
  paramsForSelection(const SelectionTree::Node *N);

  const ObjCInterfaceDecl *Interface = nullptr;

  // Will be nullptr if running on an interface.
  const ObjCImplementationDecl *Impl = nullptr;
};

REGISTER_TWEAK(ObjCMemberwiseInitializer)

bool ObjCMemberwiseInitializer::prepare(const Selection &Inputs) {
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  if (!N)
    return false;
  const Decl *D = N->ASTNode.get<Decl>();
  if (!D)
    return false;
  const auto &LangOpts = Inputs.AST->getLangOpts();
  // Require ObjC w/ arc enabled since we don't emit retains.
  if (!LangOpts.ObjC || !LangOpts.ObjCAutoRefCount)
    return false;

  // We support the following selected decls:
  // - ObjCInterfaceDecl/ObjCImplementationDecl only - generate for all
  //   properties and ivars
  //
  // - Specific ObjCPropertyDecl(s)/ObjCIvarDecl(s) - generate only for those
  //   selected. Note that if only one is selected, the common ancestor will be
  //   the ObjCPropertyDecl/ObjCIvarDecl itself instead of the container.
  if (const auto *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    // Ignore forward declarations (@class Name;).
    if (!ID->isThisDeclarationADefinition())
      return false;
    Interface = ID;
  } else if (const auto *ID = dyn_cast<ObjCImplementationDecl>(D)) {
    Interface = ID->getClassInterface();
    Impl = ID;
  } else if (isa<ObjCPropertyDecl, ObjCIvarDecl>(D)) {
    const auto *DC = D->getDeclContext();
    if (const auto *ID = dyn_cast<ObjCInterfaceDecl>(DC)) {
      Interface = ID;
    } else if (const auto *ID = dyn_cast<ObjCImplementationDecl>(DC)) {
      Interface = ID->getClassInterface();
      Impl = ID;
    }
  }
  return Interface != nullptr;
}

SmallVector<MethodParameter, 8>
ObjCMemberwiseInitializer::paramsForSelection(const SelectionTree::Node *N) {
  SmallVector<MethodParameter, 8> Params;
  // Base case: selected a single ivar or property.
  if (const auto *D = N->ASTNode.get<Decl>()) {
    if (auto Param = MethodParameter::parameterFor(*D)) {
      Params.push_back(Param.getValue());
      return Params;
    }
  }
  const ObjCContainerDecl *Container =
      Impl ? static_cast<const ObjCContainerDecl *>(Impl)
           : static_cast<const ObjCContainerDecl *>(Interface);
  if (Container == N->ASTNode.get<ObjCContainerDecl>() && N->Children.empty())
    return getAllParams(Interface);

  llvm::DenseSet<llvm::StringRef> Names;
  // Check for selecting multiple ivars/properties.
  for (const auto *CNode : N->Children) {
    const Decl *D = CNode->ASTNode.get<Decl>();
    if (!D)
      continue;
    if (auto P = MethodParameter::parameterFor(*D))
      if (Names.insert(P->Name).second)
        Params.push_back(P.getValue());
  }
  return Params;
}

Expected<Tweak::Effect>
ObjCMemberwiseInitializer::apply(const Selection &Inputs) {
  const auto &SM = Inputs.AST->getASTContext().getSourceManager();
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  if (!N)
    return error("Invalid selection");

  SmallVector<MethodParameter, 8> Params = paramsForSelection(N);

  // Insert before the first non-init instance method.
  std::vector<Anchor> Anchors = {
      {[](const Decl *D) {
         if (const auto *MD = llvm::dyn_cast<ObjCMethodDecl>(D)) {
           return MD->getMethodFamily() != OMF_init && MD->isInstanceMethod();
         }
         return false;
       },
       Anchor::Above}};
  Effect E;

  auto InterfaceReplacement =
      insertDecl(initializerForParams(Params, /*GenerateImpl=*/false),
                 *Interface, Anchors);
  if (!InterfaceReplacement)
    return InterfaceReplacement.takeError();
  auto FE = Effect::fileEdit(SM, SM.getFileID(Interface->getLocation()),
                             tooling::Replacements(*InterfaceReplacement));
  if (!FE)
    return FE.takeError();
  E.ApplyEdits.insert(std::move(*FE));

  if (Impl) {
    // If we see the class implementation, add the initializer there too.
    // FIXME: merging the edits is awkward, do this elsewhere.
    auto ImplReplacement = insertDecl(
        initializerForParams(Params, /*GenerateImpl=*/true), *Impl, Anchors);
    if (!ImplReplacement)
      return ImplReplacement.takeError();

    if (SM.isWrittenInSameFile(Interface->getLocation(), Impl->getLocation())) {
      // Merge with previous edit if they are in the same file.
      if (auto Err =
              E.ApplyEdits.begin()->second.Replacements.add(*ImplReplacement))
        return std::move(Err);
    } else {
      // Generate a new edit if the interface and implementation are in
      // different files.
      auto FE = Effect::fileEdit(SM, SM.getFileID(Impl->getLocation()),
                                 tooling::Replacements(*ImplReplacement));
      if (!FE)
        return FE.takeError();
      E.ApplyEdits.insert(std::move(*FE));
    }
  }
  return E;
}

std::string ObjCMemberwiseInitializer::title() const {
  if (Impl)
    return "Generate memberwise initializer";
  return "Declare memberwise initializer";
}

} // namespace
} // namespace clangd
} // namespace clang
