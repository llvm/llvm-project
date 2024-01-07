#ifndef SYMBOLGRAPH_H
#define SYMBOLGRAPH_H

#include "clang/Basic/LangStandard.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <vector>

namespace sgmerger {

// see https://github.com/apple/swift-docc-symbolkit/bdob/main/openapi.yaml
struct SymbolGraph {

  struct Symbol {
    Symbol(const llvm::json::Object &SymbolObj);

    llvm::json::Object SymbolObj;
    std::string AccessLevel;
    clang::extractapi::APIRecord::RecordKind Kind;
    clang::extractapi::DeclarationFragments DeclFragments;
    clang::extractapi::FunctionSignature FunctionSign;
    std::string Name;
    std::string USR;
    clang::extractapi::AvailabilitySet Availabilities;
    clang::extractapi::DocComment Comments;
    clang::extractapi::RecordLocation Location;
    clang::extractapi::DeclarationFragments SubHeadings;

    // underlying type in case of Typedef
    clang::extractapi::SymbolReference UnderLyingType;
  };

  SymbolGraph(const llvm::StringRef JSON);
  llvm::json::Object SymbolGraphObject;
  llvm::json::Object Metadata;
  llvm::json::Object Module;
  std::vector<Symbol> Symbols;
  llvm::json::Array Relationships;
};

} // namespace sgmerger

#endif /* SYMBOLGRAPH_H */
