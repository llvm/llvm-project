//===--- ASTDumperUtils.h - Printing of AST nodes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AST utilities for traversal down the tree.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTDUMPERUTILS_H
#define LLVM_CLANG_AST_ASTDUMPERUTILS_H

#include "llvm/Support/raw_ostream.h"

namespace clang {

/// Used to specify the format for printing AST dump information.
enum ASTDumpOutputFormat {
  ADOF_Default,
  ADOF_JSON
};

// Colors used for various parts of the AST dump
// Do not use bold yellow for any text.  It is hard to read on white screens.

struct TerminalColor {
  llvm::raw_ostream::Colors Color;
  bool Bold;
};

struct ASTDumpColor {
  // Red           - Cast
  // Green         - Type
  // Bold Green    - DeclKindName, Undeserialized
  // Yellow        - Address, Location
  // Blue          - Comment, Null, Indent
  // Bold Blue     - Attr
  // Bold Magenta  - Stmt
  // Cyan          - ValueKind, ObjectKind
  // Bold Cyan     - Value, DeclName

  // Decl kind names (VarDecl, FunctionDecl, etc)
  static constexpr TerminalColor DeclKindName = {llvm::raw_ostream::GREEN,
                                                 true};
  // Attr names (CleanupAttr, GuardedByAttr, etc)
  static constexpr TerminalColor Attr = {llvm::raw_ostream::BLUE, true};
  // Statement names (DeclStmt, ImplicitCastExpr, etc)
  static constexpr TerminalColor Stmt = {llvm::raw_ostream::MAGENTA, true};
  // Comment names (FullComment, ParagraphComment, TextComment, etc)
  static constexpr TerminalColor Comment = {llvm::raw_ostream::BLUE, false};

  // Type names (int, float, etc, plus user defined types)
  static constexpr TerminalColor Type = {llvm::raw_ostream::GREEN, false};

  // Pointer address
  static constexpr TerminalColor Address = {llvm::raw_ostream::YELLOW, false};
  // Source locations
  static constexpr TerminalColor Location = {llvm::raw_ostream::YELLOW, false};

  // lvalue/xvalue
  static constexpr TerminalColor ValueKind = {llvm::raw_ostream::CYAN, false};
  // bitfield/objcproperty/objcsubscript/vectorcomponent
  static constexpr TerminalColor ObjectKind = {llvm::raw_ostream::CYAN, false};
  // contains-errors
  static constexpr TerminalColor Errors = {llvm::raw_ostream::RED, true};

  // Null statements
  static constexpr TerminalColor Null = {llvm::raw_ostream::BLUE, false};

  // Undeserialized entities
  static constexpr TerminalColor Undeserialized = {llvm::raw_ostream::GREEN,
                                                   true};

  // CastKind from CastExpr's
  static constexpr TerminalColor Cast = {llvm::raw_ostream::RED, false};

  // Value of the statement
  static constexpr TerminalColor Value = {llvm::raw_ostream::CYAN, true};
  // Decl names
  static constexpr TerminalColor DeclName = {llvm::raw_ostream::CYAN, true};

  // Indents ( `, -. | )
  static constexpr TerminalColor Indent = {llvm::raw_ostream::BLUE, false};
};

class ColorScope {
  llvm::raw_ostream &OS;
  const bool ShowColors;

public:
  ColorScope(llvm::raw_ostream &OS, bool ShowColors, TerminalColor Color)
      : OS(OS), ShowColors(ShowColors) {
    if (ShowColors)
      OS.changeColor(Color.Color, Color.Bold);
  }
  ~ColorScope() {
    if (ShowColors)
      OS.resetColor();
  }
};

} // namespace clang

#endif // LLVM_CLANG_AST_ASTDUMPERUTILS_H
