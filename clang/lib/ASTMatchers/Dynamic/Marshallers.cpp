#include "Marshallers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <string>

static llvm::Optional<std::string>
getBestGuess(llvm::StringRef Search, llvm::ArrayRef<llvm::StringRef> Allowed,
             llvm::StringRef DropPrefix = "", unsigned MaxEditDistance = 3) {
  if (MaxEditDistance != ~0U)
    ++MaxEditDistance;
  llvm::StringRef Res;
  for (const llvm::StringRef &Item : Allowed) {
    if (Item.equals_lower(Search)) {
      assert(!Item.equals(Search) && "This should be handled earlier on.");
      MaxEditDistance = 1;
      Res = Item;
      continue;
    }
    unsigned Distance = Item.edit_distance(Search);
    if (Distance < MaxEditDistance) {
      MaxEditDistance = Distance;
      Res = Item;
    }
  }
  if (!Res.empty())
    return Res.str();
  if (!DropPrefix.empty()) {
    --MaxEditDistance; // Treat dropping the prefix as 1 edit
    for (const llvm::StringRef &Item : Allowed) {
      auto NoPrefix = Item;
      if (!NoPrefix.consume_front(DropPrefix))
        continue;
      if (NoPrefix.equals_lower(Search)) {
        if (NoPrefix.equals(Search))
          return Item.str();
        MaxEditDistance = 1;
        Res = Item;
        continue;
      }
      unsigned Distance = NoPrefix.edit_distance(Search);
      if (Distance < MaxEditDistance) {
        MaxEditDistance = Distance;
        Res = Item;
      }
    }
    if (!Res.empty())
      return Res.str();
  }
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::attr::Kind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define ATTR(X) "attr::" #X,
#include "clang/Basic/AttrList.inc"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "attr::");
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::CastKind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define CAST_OPERATION(Name) #Name,
#include "clang/AST/OperationKinds.def"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed));
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::OpenMPClauseKind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define OMP_CLAUSE_CLASS(Enum, Str, Class) #Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "OMPC_");
  return llvm::None;
}
