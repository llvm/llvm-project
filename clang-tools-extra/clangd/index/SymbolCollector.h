//===--- SymbolCollector.h ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"

namespace clang {
namespace clangd {

// Collect all symbols from an AST.
//
// Clients (e.g. clangd) can use SymbolCollector together with
// index::indexTopLevelDecls to retrieve all symbols when the source file is
// changed.
class SymbolCollector : public index::IndexDataConsumer {
public:
  SymbolCollector() = default;

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations, FileID FID,
                      unsigned Offset,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override;

  SymbolSlab takeSymbols() { return std::move(Symbols).build(); }

private:
  // All Symbols collected from the AST.
  SymbolSlab::Builder Symbols;
};

} // namespace clangd
} // namespace clang
