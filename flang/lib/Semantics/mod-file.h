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
#include <sstream>
#include <string>

namespace Fortran::parser {
class CharBlock;
class Message;
class MessageFixedText;
}

namespace Fortran::semantics {

using SourceName = parser::CharBlock;
class Symbol;
class Scope;
class SemanticsContext;

class ModFileWriter {
public:
  ModFileWriter(SemanticsContext &context) : context_{context} {}
  bool WriteAll();

private:
  SemanticsContext &context_;
  std::stringstream uses_;
  std::stringstream useExtraAttrs_;  // attrs added to used entity
  std::stringstream decls_;
  std::stringstream contains_;

  void WriteAll(const Scope &);
  void WriteOne(const Scope &);
  void Write(const Symbol &);
  std::string GetAsString(const Symbol &);
  void PutSymbols(const Scope &);
  void PutSymbol(std::stringstream &, const Symbol &);
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

}
#endif
