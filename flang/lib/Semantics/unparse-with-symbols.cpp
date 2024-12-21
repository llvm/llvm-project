//===-- lib/Semantics/unparse-with-symbols.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/unparse-with-symbols.h"
#include "mod-file.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

namespace Fortran::semantics {

// Walk the parse tree and collection information about which statements
// reference symbols. Then PrintSymbols outputs information by statement.
// The first reference to a symbol is treated as its definition and more
// information is included.
class SymbolDumpVisitor {
public:
  // Write out symbols referenced at this statement.
  void PrintSymbols(const parser::CharBlock &, llvm::raw_ostream &, int);

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}
  template <typename T> bool Pre(const parser::Statement<T> &stmt) {
    currStmt_ = stmt.source;
    return true;
  }
  template <typename T> void Post(const parser::Statement<T> &) {
    currStmt_ = std::nullopt;
  }
  bool Pre(const parser::AccClause &clause) {
    currStmt_ = clause.source;
    return true;
  }
  void Post(const parser::AccClause &) { currStmt_ = std::nullopt; }
  bool Pre(const parser::OmpClause &clause) {
    currStmt_ = clause.source;
    return true;
  }
  void Post(const parser::OmpClause &) { currStmt_ = std::nullopt; }
  bool Pre(const parser::OpenMPThreadprivate &dir) {
    currStmt_ = dir.source;
    return true;
  }
  void Post(const parser::OpenMPThreadprivate &) { currStmt_ = std::nullopt; }
  void Post(const parser::Name &name);

  bool Pre(const parser::OpenMPDeclareMapperConstruct &x) {
    currStmt_ = x.source;
    return true;
  }
  void Post(const parser::OpenMPDeclareMapperConstruct &) {
    currStmt_ = std::nullopt;
  }

private:
  std::optional<SourceName> currStmt_; // current statement we are processing
  std::multimap<const char *, const Symbol *> symbols_; // location to symbol
  std::set<const Symbol *> symbolsDefined_; // symbols that have been processed
  void Indent(llvm::raw_ostream &, int) const;
};

void SymbolDumpVisitor::PrintSymbols(
    const parser::CharBlock &location, llvm::raw_ostream &out, int indent) {
  std::set<const Symbol *> done; // prevent duplicates on this line
  auto range{symbols_.equal_range(location.begin())};
  for (auto it{range.first}; it != range.second; ++it) {
    const auto *symbol{it->second};
    if (done.insert(symbol).second) {
      bool firstTime{symbolsDefined_.insert(symbol).second};
      Indent(out, indent);
      out << '!' << (firstTime ? "DEF"s : "REF"s) << ": ";
      DumpForUnparse(out, *symbol, firstTime);
      out << '\n';
    }
  }
}

void SymbolDumpVisitor::Indent(llvm::raw_ostream &out, int indent) const {
  for (int i{0}; i < indent; ++i) {
    out << ' ';
  }
}

void SymbolDumpVisitor::Post(const parser::Name &name) {
  if (const auto *symbol{name.symbol}) {
    if (!symbol->has<MiscDetails>()) {
      symbols_.emplace(currStmt_.value().begin(), symbol);
    }
  }
}

void UnparseWithSymbols(llvm::raw_ostream &out, const parser::Program &program,
    parser::Encoding encoding) {
  SymbolDumpVisitor visitor;
  parser::Walk(program, visitor);
  parser::preStatementType preStatement{
      [&](const parser::CharBlock &location, llvm::raw_ostream &out,
          int indent) { visitor.PrintSymbols(location, out, indent); }};
  parser::Unparse(out, program, encoding, false, true, &preStatement);
}

// UnparseWithModules()

class UsedModuleVisitor {
public:
  UnorderedSymbolSet &modulesUsed() { return modulesUsed_; }
  UnorderedSymbolSet &modulesDefined() { return modulesDefined_; }
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}
  void Post(const parser::ModuleStmt &module) {
    if (module.v.symbol) {
      modulesDefined_.insert(*module.v.symbol);
    }
  }
  void Post(const parser::UseStmt &use) {
    if (use.moduleName.symbol) {
      modulesUsed_.insert(*use.moduleName.symbol);
    }
  }

private:
  UnorderedSymbolSet modulesUsed_;
  UnorderedSymbolSet modulesDefined_;
};

void UnparseWithModules(llvm::raw_ostream &out, SemanticsContext &context,
    const parser::Program &program, parser::Encoding encoding) {
  UsedModuleVisitor visitor;
  parser::Walk(program, visitor);
  UnorderedSymbolSet nonIntrinsicModulesWritten{
      std::move(visitor.modulesDefined())};
  ModFileWriter writer{context};
  for (SymbolRef moduleRef : visitor.modulesUsed()) {
    writer.WriteClosure(out, *moduleRef, nonIntrinsicModulesWritten);
  }
  parser::Unparse(out, program, encoding, false, true);
}
} // namespace Fortran::semantics
