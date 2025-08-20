//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/HLSL/Frontend/FrontendActions.h"
#include "clang/Parse/ParseHLSLRootSignature.h"
#include "clang/Sema/Sema.h"

namespace clang {

class InjectRootSignatureCallback : public PPCallbacks {
private:
  Sema &Actions;
  StringRef RootSigName;
  llvm::dxbc::RootSignatureVersion Version;

  std::optional<StringLiteral *> processStringLiteral(ArrayRef<Token> Tokens) {
    for (Token Tok : Tokens)
      if (!tok::isStringLiteral(Tok.getKind()))
        return std::nullopt;

    ExprResult StringResult = Actions.ActOnUnevaluatedStringLiteral(Tokens);
    if (StringResult.isInvalid())
      return std::nullopt;

    if (auto Signature = dyn_cast<StringLiteral>(StringResult.get()))
      return Signature;

    return std::nullopt;
  }

public:
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (RootSigName != MacroNameTok.getIdentifierInfo()->getName())
      return;

    const MacroInfo *MI = MD->getMacroInfo();
    auto Signature = processStringLiteral(MI->tokens());
    if (!Signature.has_value()) {
      Actions.getDiagnostics().Report(MI->getDefinitionLoc(),
                                      diag::err_expected_string_literal)
          << /*in attributes...*/ 4 << "RootSignature";
      return;
    }

    IdentifierInfo *DeclIdent =
        hlsl::ParseHLSLRootSignature(Actions, Version, *Signature);
  }

  InjectRootSignatureCallback(Sema &Actions, StringRef RootSigName,
                              llvm::dxbc::RootSignatureVersion Version)
      : PPCallbacks(), Actions(Actions), RootSigName(RootSigName),
        Version(Version) {}
};

void HLSLFrontendAction::ExecuteAction() {
  // Pre-requisites to invoke
  CompilerInstance &CI = getCompilerInstance();
  if (!CI.hasASTContext() || !CI.hasPreprocessor())
    return WrapperFrontendAction::ExecuteAction();

  // InjectRootSignatureCallback requires access to invoke Sema to lookup/
  // register a root signature declaration. The wrapped action is required to
  // account for this by only creating a Sema if one doesn't already exist
  // (like we have done, and, ASTFrontendAction::ExecuteAction)
  if (!CI.hasSema())
    CI.createSema(getTranslationUnitKind(),
                  /*CodeCompleteConsumer=*/nullptr);
  Sema &S = CI.getSema();

  // Register HLSL specific callbacks
  auto LangOpts = CI.getLangOpts();
  auto MacroCallback = std::make_unique<InjectRootSignatureCallback>(
      S, LangOpts.HLSLRootSigOverride, LangOpts.HLSLRootSigVer);

  Preprocessor &PP = CI.getPreprocessor();
  PP.addPPCallbacks(std::move(MacroCallback));

  // Invoke as normal
  WrapperFrontendAction::ExecuteAction();
}

HLSLFrontendAction::HLSLFrontendAction(
    std::unique_ptr<FrontendAction> WrappedAction)
    : WrapperFrontendAction(std::move(WrappedAction)) {}

} // namespace clang
