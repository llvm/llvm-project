//===--- ASTDumperUtils.h - Printing of AST nodes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AST utilities for traversal down the tree.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTDUMPERUTILS_H
#define LLVM_CLANG_AST_ASTDUMPERUTILS_H

#include "clang/AST/ASTContext.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

// Colors used for various parts of the AST dump
// Do not use bold yellow for any text.  It is hard to read on white screens.

struct TerminalColor {
  raw_ostream::Colors Color;
  bool Bold;
};

// Red           - CastColor
// Green         - TypeColor
// Bold Green    - DeclKindNameColor, UndeserializedColor
// Yellow        - AddressColor, LocationColor
// Blue          - CommentColor, NullColor, IndentColor
// Bold Blue     - AttrColor
// Bold Magenta  - StmtColor
// Cyan          - ValueKindColor, ObjectKindColor
// Bold Cyan     - ValueColor, DeclNameColor

// Decl kind names (VarDecl, FunctionDecl, etc)
static const TerminalColor DeclKindNameColor = {raw_ostream::GREEN, true};
// Attr names (CleanupAttr, GuardedByAttr, etc)
static const TerminalColor AttrColor = {raw_ostream::BLUE, true};
// Statement names (DeclStmt, ImplicitCastExpr, etc)
static const TerminalColor StmtColor = {raw_ostream::MAGENTA, true};
// Comment names (FullComment, ParagraphComment, TextComment, etc)
static const TerminalColor CommentColor = {raw_ostream::BLUE, false};

// Type names (int, float, etc, plus user defined types)
static const TerminalColor TypeColor = {raw_ostream::GREEN, false};

// Pointer address
static const TerminalColor AddressColor = {raw_ostream::YELLOW, false};
// Source locations
static const TerminalColor LocationColor = {raw_ostream::YELLOW, false};

// lvalue/xvalue
static const TerminalColor ValueKindColor = {raw_ostream::CYAN, false};
// bitfield/objcproperty/objcsubscript/vectorcomponent
static const TerminalColor ObjectKindColor = {raw_ostream::CYAN, false};

// Null statements
static const TerminalColor NullColor = {raw_ostream::BLUE, false};

// Undeserialized entities
static const TerminalColor UndeserializedColor = {raw_ostream::GREEN, true};

// CastKind from CastExpr's
static const TerminalColor CastColor = {raw_ostream::RED, false};

// Value of the statement
static const TerminalColor ValueColor = {raw_ostream::CYAN, true};
// Decl names
static const TerminalColor DeclNameColor = {raw_ostream::CYAN, true};

// Indents ( `, -. | )
static const TerminalColor IndentColor = {raw_ostream::BLUE, false};

class ColorScope {
  raw_ostream &OS;
  const bool ShowColors;

public:
  ColorScope(raw_ostream &OS, bool ShowColors, TerminalColor Color)
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
