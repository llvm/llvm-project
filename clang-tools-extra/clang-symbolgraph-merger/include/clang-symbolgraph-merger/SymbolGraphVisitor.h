#ifndef SYMBOLGRAPHVISITOR_H
#define SYMBOLGRAPHVISITOR_H

#include "clang-symbolgraph-merger/SymbolGraph.h"
#include "clang/ExtractAPI/API.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace sgmerger {

// Visits a symbol graph obbect and record the extracted info to API
template <typename Derived> class SymbolGraphVisitor {
public:
  bool traverseSymbolGraph(const SymbolGraph &SG) {
    bool Success = true;
    Success = (getDerived()->visitMetadata(SG.Metadata) &&
               getDerived()->visitModule(SG.Module) &&
               getDerived()->traverseSymbols(SG.Symbols) &&
               getDerived()->traverseRelationships(SG.Relationships));

    return Success;
  }

  bool traverseSymbols(const std::vector<SymbolGraph::Symbol> &Symbols) {
    bool Success = true;
    for (const auto &Symbol : Symbols)
      Success = getDerived()->visitSymbol(Symbol);
    return Success;
  }

  bool traverseRelationships(const llvm::json::Array &Relationships) {
    bool Success = true;
    for (const auto &RelValue : Relationships) {
      if (const auto *RelObj = RelValue.getAsObject())
        Success = getDerived()->visitRelationship(*RelObj);
    }
    return Success;
  }

  bool visitMetadata(const llvm::json::Object &Metadata);
  bool visitModule(const llvm::json::Object &Module);
  bool visitSymbol(const SymbolGraph::Symbol &Symbol);
  bool visitRelationship(const llvm::json::Object &Relationship);

  std::unique_ptr<clang::extractapi::APISet> getAPISet() {
    return std::move(API);
  }

protected:
  std::unique_ptr<clang::extractapi::APISet> API;

public:
  SymbolGraphVisitor(const SymbolGraphVisitor &) = delete;
  SymbolGraphVisitor(SymbolGraphVisitor &&) = delete;
  SymbolGraphVisitor &operator=(const SymbolGraphVisitor &) = delete;
  SymbolGraphVisitor &operator=(SymbolGraphVisitor &&) = delete;

protected:
  SymbolGraphVisitor() : API(nullptr) {}
  ~SymbolGraphVisitor() = default;

  Derived *getDerived() { return static_cast<Derived *>(this); };
};

} // namespace sgmerger

#endif /* SYMBOLGRAPHVISITOR_H */
