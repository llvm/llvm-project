//===- ScriptLexer.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the nums for LinkerScript lexer
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SCRIPT_TOKENIZER_H
#define LLD_ELF_SCRIPT_TOKENIZER_H

namespace lld {
namespace elf {
enum class Tok {
  Entry,

  // Commands Files
  Include,
  Input,
  Group,
  Memory,
  Output,
  SearchDir,
  Startup,

  Insert,
  After,

  // Commands for object file formats
  OutputFormat,
  Target,

  // Other linker script commands
  Assert,
  Constant,
  Extern,
  // FORCE_COMMON_ALLOCATION
  // INHIBIT_COMMON_ALLOCATION
  OutputArch,
  Nocrossrefs,
  NocrossrefsTo,

  // Assignment
  Provide,
  Hidden,
  ProvideHidden,

  Sections,
  Before,

  // Input Section
  ExcludeFile,
  Keep,
  InputSectionFlags,

  // Read section
  Overlay,
  Noload,
  Copy,
  Info,

  // Output Section
  OverwriteSections,
  Subalign,
  OnlyIfRo,
  OnlyIfRw,
  Fill,
  Sort,

  // Builtin Functions
  Absolute,
  Addr,
  Align,
  Alignof,
  // BLOCK, // synonym for ALIGN for compatibility with older linker script
  DataSegmentAlign,
  DataSegmentEnd,
  DataSegmentRelroEnd,
  Defined,
  Length,
  Loadaddr,

  Log2ceil,
  Max,
  Min,
  Origin,
  SegmentStart,
  // NEXT, // This function is closely related to ALIGN(exp); unless you use the
  // MEMORY command to define discontinuous memory for the output file, the two
  // functions are equivalent.
  Sizeof,
  SizeofHeaders,

  // PHDRS Command
  Filehdr,
  Phdrs,
  At,
  Flags,

  // Version Command
  Version,

  RegionAlias,
  AsNeeded,
  Constructors,

  // Symbolic Constants
  Maxpagesize,
  Commonpagesize,

  Error,
  Eof,

  Identifier,
  Hexdecimal,  // 0x
  HexdecimalH, // end with H/h
  Decimal,
  DecimalK, // end with K/k
  DecimalM, // end with M/m

  // Symbol tokens
  LeftCurlyBracket,  // {
  RightCurlyBracket, // }
  LeftParenthesis,   // (
  RightParenthesis,  // )
  Comma,             // ,
  Semicolon,         // ;
  Colon,             // :
  Asterisk,          // *
  Question,          // ?
  Excalamation,      // !
  Backslash,         // "\"
  Slash,             // /
  Percent,           // %
  Greater,           // >
  Less,              // <
  Minus,             // -
  Plus,              // +
  BitwiseAnd,        // &
  BitwiseXor,        // ^
  BitwiseOr,         // |
  Underscore,        // _
  Dot,               // .
  Quote, // Quoted token. Note that double-quote characters are parts of a token
  // because, in a glob match context, only unquoted tokens are interpreted as
  // glob patterns. Double-quoted tokens are literal patterns in that context.

  // Assignmemnt
  Assign,           // =
  PlusAssign,       // +=
  MinusAssign,      // -=
  MulAssign,        // *=
  DivAssign,        // /=
  LeftShiftAssign,  // <<=
  RightShiftAssign, // >>=
  AndAssign,        // &=
  OrAssign,         // |=
  XorAssign,        // ^=

  // operator token
  NotEqual,     // !=
  Equal,        // ==
  GreaterEqual, // >=
  LessEqual,    // <=
  LeftShift,    // <<
  RightShift,   // >>
  LogicalAnd,   // &&
  LogicalOr     // ||
};
} // namespace elf
} // namespace lld

#endif // LLD_ELF_SCRIPT_TOKENIZER_H
