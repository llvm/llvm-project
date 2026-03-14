//===--- DumpAST.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines a few tweaks that expose AST and related information.
// Some of these are fairly clang-specific and hidden (e.g. textual AST dumps).
// Others are more generally useful (class layout) and are exposed by default.
//===----------------------------------------------------------------------===//
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Type.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

/// Dumps the AST of the selected node.
/// Input:
///   fcall("foo");
///   ^^^^^
/// Message:
///   CallExpr
///   |-DeclRefExpr fcall
///   `-StringLiteral "foo"
class DumpAST : public Tweak {
public:
  const char *id() const final;

  bool prepare(const Selection &Inputs) override {
    for (auto *N = Inputs.ASTSelection.commonAncestor(); N && !Node;
         N = N->Parent)
      if (dumpable(N->ASTNode))
        Node = N->ASTNode;
    return Node.has_value();
  }
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override {
    return std::string(
        llvm::formatv("Dump {0} AST", Node->getNodeKind().asStringRef()));
  }
  llvm::StringLiteral kind() const override { return CodeAction::INFO_KIND; }
  bool hidden() const override { return true; }

private:
  static bool dumpable(const DynTypedNode &N) {
    // Sadly not all node types can be dumped, and there's no API to check.
    // See DynTypedNode::dump().
    return N.get<Decl>() || N.get<Stmt>() || N.get<Type>();
  }

  std::optional<DynTypedNode> Node;
};
REGISTER_TWEAK(DumpAST)

llvm::Expected<Tweak::Effect> DumpAST::apply(const Selection &Inputs) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  Node->dump(OS, Inputs.AST->getASTContext());
  return Effect::showMessage(std::move(OS.str()));
}

/// Dumps the SelectionTree.
/// Input:
/// int fcall(int);
/// void foo() {
///   fcall(2 + 2);
///     ^^^^^
/// }
/// Message:
/// TranslationUnitDecl
///   FunctionDecl void foo()
///     CompoundStmt {}
///      .CallExpr fcall(2 + 2)
///        ImplicitCastExpr fcall
///         .DeclRefExpr fcall
///        BinaryOperator 2 + 2
///          *IntegerLiteral 2
class ShowSelectionTree : public Tweak {
public:
  const char *id() const final;

  bool prepare(const Selection &Inputs) override { return true; }
  Expected<Effect> apply(const Selection &Inputs) override {
    return Effect::showMessage(llvm::to_string(Inputs.ASTSelection));
  }
  std::string title() const override { return "Show selection tree"; }
  llvm::StringLiteral kind() const override { return CodeAction::INFO_KIND; }
  bool hidden() const override { return true; }
};
REGISTER_TWEAK(ShowSelectionTree)

/// Dumps the symbol under the cursor.
/// Inputs:
/// void foo();
///      ^^^
/// Message:
///  foo -
///  {"containerName":null,"id":"CA2EBE44A1D76D2A","name":"foo","usr":"c:@F@foo#"}
class DumpSymbol : public Tweak {
  const char *id() const final;
  bool prepare(const Selection &Inputs) override { return true; }
  Expected<Effect> apply(const Selection &Inputs) override {
    std::string Storage;
    llvm::raw_string_ostream Out(Storage);

    for (auto &Sym : getSymbolInfo(
             *Inputs.AST, sourceLocToPosition(Inputs.AST->getSourceManager(),
                                              Inputs.Cursor)))
      Out << Sym;
    return Effect::showMessage(Out.str());
  }
  std::string title() const override { return "Dump symbol under the cursor"; }
  llvm::StringLiteral kind() const override { return CodeAction::INFO_KIND; }
  bool hidden() const override { return true; }
};
REGISTER_TWEAK(DumpSymbol)

/// Shows the layout of the RecordDecl under the cursor.
/// Input:
/// struct X { int foo; };
/// ^^^^^^^^
/// Message:
///        0 | struct S
///        0 |   int foo
///          | [sizeof=4, dsize=4, align=4,
///          |  nvsize=4, nvalign=4]
class DumpRecordLayout : public Tweak {
public:
  const char *id() const final;

  bool prepare(const Selection &Inputs) override {
    if (auto *Node = Inputs.ASTSelection.commonAncestor())
      if (auto *D = Node->ASTNode.get<Decl>())
        Record = dyn_cast<RecordDecl>(D);
    return Record && Record->isThisDeclarationADefinition() &&
           !Record->isDependentType();
  }
  Expected<Effect> apply(const Selection &Inputs) override {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    Inputs.AST->getASTContext().DumpRecordLayout(Record, OS);
    return Effect::showMessage(std::move(OS.str()));
  }
  std::string title() const override {
    return std::string(llvm::formatv(
        "Show {0} layout",
        TypeWithKeyword::getTagTypeKindName(Record->getTagKind())));
  }
  llvm::StringLiteral kind() const override { return CodeAction::INFO_KIND; }
  // FIXME: this is interesting to most users. However:
  //  - triggering is too broad (e.g. triggers on comments within a class)
  //  - showMessage has inconsistent UX (e.g. newlines are stripped in VSCode)
  //  - the output itself is a bit hard to decipher.
  bool hidden() const override { return true; }

private:
  const RecordDecl *Record = nullptr;
};
REGISTER_TWEAK(DumpRecordLayout)

} // namespace
} // namespace clangd
} // namespace clang
