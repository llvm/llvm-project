//===-- lib/Semantics/mod-file.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_MOD_FILE_H_
#define FORTRAN_SEMANTICS_MOD_FILE_H_

#include "flang/Semantics/attr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace Fortran::parser {
class CharBlock;
class Message;
class MessageFixedText;
} // namespace Fortran::parser

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {

using SourceName = parser::CharBlock;
class Symbol;
class Scope;
class SemanticsContext;

class ModFileWriter {
public:
  explicit ModFileWriter(SemanticsContext &context) : context_{context} {}
  bool WriteAll();

private:
  SemanticsContext &context_;
  // Buffer to use with raw_string_ostream
  std::string usesBuf_;
  std::string useExtraAttrsBuf_;
  std::string declsBuf_;
  std::string containsBuf_;

  llvm::raw_string_ostream uses_{usesBuf_};
  llvm::raw_string_ostream useExtraAttrs_{
      useExtraAttrsBuf_}; // attrs added to used entity
  llvm::raw_string_ostream decls_{declsBuf_};
  llvm::raw_string_ostream contains_{containsBuf_};

  void WriteAll(const Scope &);
  void WriteOne(const Scope &);
  void Write(const Symbol &);
  std::string GetAsString(const Symbol &);
  // Returns true if a derived type with bindings and "contains" was emitted
  bool PutSymbols(const Scope &);
  void PutSymbol(llvm::raw_ostream &, const Symbol &);
  void PutDerivedType(const Symbol &);
  void PutSubprogram(const Symbol &);
  void PutGeneric(const Symbol &);
  void PutUse(const Symbol &);
  void PutUseExtraAttr(Attr, const Symbol &, const Symbol &);
};

class ModFileReader {
public:
  // directories specifies where to search for module files
  ModFileReader(SemanticsContext &context) : context_{context} {}
  // Find and read the module file for a module or submodule.
  // If ancestor is specified, look for a submodule of that module.
  // Return the Scope for that module/submodule or nullptr on error.
  Scope *Read(const SourceName &, Scope *ancestor = nullptr);

private:
  SemanticsContext &context_;

  parser::Message &Say(const SourceName &, const std::string &,
      parser::MessageFixedText &&, const std::string &);
};

} // namespace Fortran::semantics
#endif
