#ifndef SYMBOLGRAPHMERGER_H
#define SYMBOLGRAPHMERGER_H

#include "clang-symbolgraph-merger/SymbolGraph.h"
#include "clang-symbolgraph-merger/SymbolGraphVisitor.h"
#include "clang/Basic/LangStandard.h"
#include "clang/ExtractAPI/API.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>

namespace sgmerger {

using SymbolMap = llvm::DenseMap<llvm::StringRef, const SymbolGraph::Symbol *>;

class SymbolGraphMerger : public SymbolGraphVisitor<SymbolGraphMerger> {
public:
  SymbolGraphMerger(const clang::SmallVector<SymbolGraph> &SymbolGraphs,
                    const std::string &ProductName = "")
      : ProductName(ProductName), Lang(clang::Language::Unknown),
        SymbolGraphs(SymbolGraphs) {}
  bool merge();
  bool visitMetadata(const llvm::json::Object &Metadata);
  bool visitModule(const llvm::json::Object &Module);
  bool visitSymbol(const SymbolGraph::Symbol &Symbol);
  bool visitRelationship(const llvm::json::Object &Relationship);

private:
  std::string Generator;

  // stuff required to construct the APISet
  std::string ProductName;
  llvm::Triple Target;
  clang::Language Lang;

  SymbolMap PendingSymbols;
  SymbolMap VisitedSymbols;

  const clang::SmallVector<SymbolGraph> &SymbolGraphs;
};

} // namespace sgmerger

#endif /* SYMBOLGRAPHMERGER_H */
