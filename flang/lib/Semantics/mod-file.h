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
#include "flang/Semantics/symbol.h"
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
  // Tracks nested DEC structures and fields of that type
  UnorderedSymbolSet emittedDECStructures_, emittedDECFields_;

  llvm::raw_string_ostream uses_{usesBuf_};
  llvm::raw_string_ostream useExtraAttrs_{
      useExtraAttrsBuf_}; // attrs added to used entity
  llvm::raw_string_ostream decls_{declsBuf_};
  llvm::raw_string_ostream contains_{containsBuf_};
  bool isSubmodule_{false};

  void WriteAll(const Scope &);
  void WriteOne(const Scope &);
  void Write(const Symbol &);
  std::string GetAsString(const Symbol &);
  void PutSymbols(const Scope &);
  // Returns true if a derived type with bindings and "contains" was emitted
  bool PutComponents(const Symbol &);
  void PutSymbol(llvm::raw_ostream &, const Symbol &);
  void PutEntity(llvm::raw_ostream &, const Symbol &);
  void PutEntity(
      llvm::raw_ostream &, const Symbol &, std::function<void()>, Attrs);
  void PutObjectEntity(llvm::raw_ostream &, const Symbol &);
  void PutProcEntity(llvm::raw_ostream &, const Symbol &);
  void PutDerivedType(const Symbol &, const Scope * = nullptr);
  void PutDECStructure(const Symbol &, const Scope * = nullptr);
  void PutTypeParam(llvm::raw_ostream &, const Symbol &);
  void PutSubprogram(const Symbol &);
  void PutGeneric(const Symbol &);
  void PutUse(const Symbol &);
  void PutUseExtraAttr(Attr, const Symbol &, const Symbol &);
  llvm::raw_ostream &PutAttrs(llvm::raw_ostream &, Attrs,
      const std::string * = nullptr, bool = false, std::string before = ","s,
      std::string after = ""s) const;
  void PutDirective(llvm::raw_ostream &, const Symbol &);
};

class ModFileReader {
public:
  // directories specifies where to search for module files
  ModFileReader(SemanticsContext &context) : context_{context} {}
  // Find and read the module file for a module or submodule.
  // If ancestor is specified, look for a submodule of that module.
  // Return the Scope for that module/submodule or nullptr on error.
  Scope *Read(const SourceName &, std::optional<bool> isIntrinsic,
      Scope *ancestor, bool silent = false);

private:
  SemanticsContext &context_;

  parser::Message &Say(const SourceName &, const std::string &,
      parser::MessageFixedText &&, const std::string &);
};

} // namespace Fortran::semantics
#endif
