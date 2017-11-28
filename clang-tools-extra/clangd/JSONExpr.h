//===--- JSONExpr.h - JSON expressions, parsing and serialization - C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

// FIXME: rename to JSON.h now that the scope is wider?

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSON_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSON_H

#include <map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace json {

// An Expr is an JSON value of unknown type.
// They can be copied, but should generally be moved.
//
// === Composing expressions ===
//
// You can implicitly construct Exprs from:
//   - strings: std::string, SmallString, formatv, StringRef, char*
//              (char*, and StringRef are references, not copies!)
//   - numbers
//   - booleans
//   - null: nullptr
//   - arrays: {"foo", 42.0, false}
//   - serializable things: any T with a T::unparse(const T&) -> Expr
//
// They can also be constructed from object/array helpers:
//   - json::obj is a type like map<StringExpr, Expr>
//   - json::ary is a type like vector<Expr>
// These can be list-initialized, or used to build up collections in a loop.
// json::ary(Collection) converts all items in a collection to Exprs.
//
// === Inspecting expressions ===
//
// Each Expr is one of the JSON kinds:
//   null    (nullptr_t)
//   boolean (bool)
//   number  (double)
//   string  (StringRef)
//   array   (json::ary)
//   object  (json::obj)
//
// The kind can be queried directly, or implicitly via the typed accessors:
//   if (Optional<StringRef> S = E.asString()
//     assert(E.kind() == Expr::String);
//
// Array and Object also have typed indexing accessors for easy traversal:
//   Expected<Expr> E = parse(R"( {"options": {"font": "sans-serif"}} )");
//   if (json::obj* O = E->asObject())
//     if (json::obj* Opts = O->getObject("options"))
//       if (Optional<StringRef> Font = Opts->getString("font"))
//         assert(Opts->at("font").kind() == Expr::String);
//
// === Serialization ===
//
// Exprs can be serialized to JSON:
//   1) raw_ostream << Expr                    // Basic formatting.
//   2) raw_ostream << formatv("{0}", Expr)    // Basic formatting.
//   3) raw_ostream << formatv("{0:2}", Expr)  // Pretty-print with indent 2.
//
// And parsed:
//   Expected<Expr> E = json::parse("[1, 2, null]");
//   assert(E && E->kind() == Expr::Array);
class Expr {
public:
  enum Kind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
  };
  class ObjectExpr;
  class ObjectKey;
  class ArrayExpr;

  // It would be nice to have Expr() be null. But that would make {} null too...
  Expr(const Expr &M) { copyFrom(M); }
  Expr(Expr &&M) { moveFrom(std::move(M)); }
  // "cheating" move-constructor for moving from initializer_list.
  Expr(const Expr &&M) { moveFrom(std::move(M)); }
  Expr(std::initializer_list<Expr> Elements) : Expr(ArrayExpr(Elements)) {}
  Expr(ArrayExpr &&Elements) : Type(T_Array) {
    create<ArrayExpr>(std::move(Elements));
  }
  Expr(ObjectExpr &&Properties) : Type(T_Object) {
    create<ObjectExpr>(std::move(Properties));
  }
  // Strings: types with value semantics.
  Expr(std::string &&V) : Type(T_String) { create<std::string>(std::move(V)); }
  Expr(const std::string &V) : Type(T_String) { create<std::string>(V); }
  Expr(const llvm::SmallVectorImpl<char> &V) : Type(T_String) {
    create<std::string>(V.begin(), V.end());
  }
  Expr(const llvm::formatv_object_base &V) : Expr(V.str()){};
  // Strings: types with reference semantics.
  Expr(llvm::StringRef V) : Type(T_StringRef) { create<llvm::StringRef>(V); }
  Expr(const char *V) : Type(T_StringRef) { create<llvm::StringRef>(V); }
  Expr(std::nullptr_t) : Type(T_Null) {}
  // Prevent implicit conversions to boolean.
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, bool>::value>::type>
  Expr(T B) : Type(T_Boolean) {
    create<bool>(B);
  }
  // Numbers: arithmetic types that are not boolean.
  template <
      typename T,
      typename = typename std::enable_if<std::is_arithmetic<T>::value>::type,
      typename = typename std::enable_if<std::integral_constant<
          bool, !std::is_same<T, bool>::value>::value>::type>
  Expr(T D) : Type(T_Number) {
    create<double>(D);
  }
  // Types with a static T::unparse function returning an Expr.
  // FIXME: should this be a free unparse() function found by ADL?
  template <typename T,
            typename = typename std::enable_if<std::is_same<
                Expr, decltype(T::unparse(*(const T *)nullptr))>::value>>
  Expr(const T &V) : Expr(T::unparse(V)) {}

  Expr &operator=(const Expr &M) {
    destroy();
    copyFrom(M);
    return *this;
  }
  Expr &operator=(Expr &&M) {
    destroy();
    moveFrom(std::move(M));
    return *this;
  }
  ~Expr() { destroy(); }

  Kind kind() const {
    switch (Type) {
    case T_Null:
      return Null;
    case T_Boolean:
      return Boolean;
    case T_Number:
      return Number;
    case T_String:
    case T_StringRef:
      return String;
    case T_Object:
      return Object;
    case T_Array:
      return Array;
    }
    llvm_unreachable("Unknown kind");
  }

  // Typed accessors return None/nullptr if the Expr is not of this type.
  llvm::Optional<std::nullptr_t> asNull() const {
    if (LLVM_LIKELY(Type == T_Null))
      return nullptr;
    return llvm::None;
  }
  llvm::Optional<bool> asBoolean() const {
    if (LLVM_LIKELY(Type == T_Boolean))
      return as<bool>();
    return llvm::None;
  }
  llvm::Optional<double> asNumber() const {
    if (LLVM_LIKELY(Type == T_Number))
      return as<double>();
    return llvm::None;
  }
  llvm::Optional<llvm::StringRef> asString() const {
    if (Type == T_String)
      return llvm::StringRef(as<std::string>());
    if (LLVM_LIKELY(Type == T_StringRef))
      return as<llvm::StringRef>();
    return llvm::None;
  }
  const ObjectExpr *asObject() const {
    return LLVM_LIKELY(Type == T_Object) ? &as<ObjectExpr>() : nullptr;
  }
  ObjectExpr *asObject() {
    return LLVM_LIKELY(Type == T_Object) ? &as<ObjectExpr>() : nullptr;
  }
  const ArrayExpr *asArray() const {
    return LLVM_LIKELY(Type == T_Array) ? &as<ArrayExpr>() : nullptr;
  }
  ArrayExpr *asArray() {
    return LLVM_LIKELY(Type == T_Array) ? &as<ArrayExpr>() : nullptr;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Expr &);

private:
  void destroy();
  void copyFrom(const Expr &M);
  // We allow moving from *const* Exprs, by marking all members as mutable!
  // This hack is needed to support initializer-list syntax efficiently.
  // (std::initializer_list<T> is a container of const T).
  void moveFrom(const Expr &&M);

  template <typename T, typename... U> void create(U &&... V) {
    new (&as<T>()) T(std::forward<U>(V)...);
  }
  template <typename T> T &as() const {
    return *reinterpret_cast<T *>(Union.buffer);
  }

  template <typename Indenter>
  void print(llvm::raw_ostream &, const Indenter &) const;
  friend struct llvm::format_provider<clang::clangd::json::Expr>;

  enum ExprType : char {
    T_Null,
    T_Boolean,
    T_Number,
    T_StringRef,
    T_String,
    T_Object,
    T_Array,
  };
  mutable ExprType Type;

public:
  // ObjectKey is a used to capture keys in Expr::ObjectExpr. Like Expr but:
  //   - only strings are allowed
  //   - it's optimized for the string literal case (Owned == nullptr)
  class ObjectKey {
  public:
    ObjectKey(const char *S) : Data(S) {}
    ObjectKey(llvm::StringRef S) : Data(S) {}
    ObjectKey(std::string &&V)
        : Owned(new std::string(std::move(V))), Data(*Owned) {}
    ObjectKey(const std::string &V) : Owned(new std::string(V)), Data(*Owned) {}
    ObjectKey(const llvm::SmallVectorImpl<char> &V)
        : ObjectKey(std::string(V.begin(), V.end())) {}
    ObjectKey(const llvm::formatv_object_base &V) : ObjectKey(V.str()) {}

    ObjectKey(const ObjectKey &C) { *this = C; }
    ObjectKey(ObjectKey &&C) : ObjectKey(static_cast<const ObjectKey &&>(C)) {}
    ObjectKey &operator=(const ObjectKey &C) {
      if (C.Owned) {
        Owned.reset(new std::string(*C.Owned));
        Data = *Owned;
      } else {
        Data = C.Data;
      }
      return *this;
    }
    ObjectKey &operator=(ObjectKey &&) = default;

    operator llvm::StringRef() const { return Data; }

    friend bool operator<(const ObjectKey &L, const ObjectKey &R) {
      return L.Data < R.Data;
    }

    // "cheating" move-constructor for moving from initializer_list.
    ObjectKey(const ObjectKey &&V) {
      Owned = std::move(V.Owned);
      Data = V.Data;
    }

  private:
    mutable std::unique_ptr<std::string> Owned; // mutable for cheating.
    llvm::StringRef Data;
  };

  class ObjectExpr : public std::map<ObjectKey, Expr> {
  public:
    explicit ObjectExpr() {}
    // Use a custom struct for list-init, because pair forces extra copies.
    struct KV;
    explicit ObjectExpr(std::initializer_list<KV> Properties);

    // Allow [] as if Expr was default-constructible as null.
    Expr &operator[](const ObjectKey &K) {
      return emplace(K, Expr(nullptr)).first->second;
    }
    Expr &operator[](ObjectKey &&K) {
      return emplace(std::move(K), Expr(nullptr)).first->second;
    }

    // Look up a property, returning nullptr if it doesn't exist.
    json::Expr *get(const ObjectKey &K) {
      auto I = find(K);
      if (I == end())
        return nullptr;
      return &I->second;
    }
    const json::Expr *get(const ObjectKey &K) const {
      auto I = find(K);
      if (I == end())
        return nullptr;
      return &I->second;
    }
    // Typed accessors return None/nullptr if
    //   - the property doesn't exist
    //   - or it has the wrong type
    llvm::Optional<std::nullptr_t> getNull(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asNull();
      return llvm::None;
    }
    llvm::Optional<bool> getBoolean(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asBoolean();
      return llvm::None;
    }
    llvm::Optional<double> getNumber(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asNumber();
      return llvm::None;
    }
    llvm::Optional<llvm::StringRef> getString(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asString();
      return llvm::None;
    }
    const ObjectExpr *getObject(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asObject();
      return nullptr;
    }
    ObjectExpr *getObject(const ObjectKey &K) {
      if (auto *V = get(K))
        return V->asObject();
      return nullptr;
    }
    const ArrayExpr *getArray(const ObjectKey &K) const {
      if (auto *V = get(K))
        return V->asArray();
      return nullptr;
    }
    ArrayExpr *getArray(const ObjectKey &K) {
      if (auto *V = get(K))
        return V->asArray();
      return nullptr;
    }
  };

  class ArrayExpr : public std::vector<Expr> {
  public:
    explicit ArrayExpr() {}
    explicit ArrayExpr(std::initializer_list<Expr> Elements) {
      reserve(Elements.size());
      for (const Expr &V : Elements)
        emplace_back(std::move(V));
    };
    template <typename Collection> explicit ArrayExpr(const Collection &C) {
      for (const auto &V : C)
        emplace_back(V);
    }

    // Typed accessors return None/nullptr if the element has the wrong type.
    llvm::Optional<std::nullptr_t> getNull(size_t I) const {
      return (*this)[I].asNull();
    }
    llvm::Optional<bool> getBoolean(size_t I) const {
      return (*this)[I].asBoolean();
    }
    llvm::Optional<double> getNumber(size_t I) const {
      return (*this)[I].asNumber();
    }
    llvm::Optional<llvm::StringRef> getString(size_t I) const {
      return (*this)[I].asString();
    }
    const ObjectExpr *getObject(size_t I) const {
      return (*this)[I].asObject();
    }
    ObjectExpr *getObject(size_t I) { return (*this)[I].asObject(); }
    const ArrayExpr *getArray(size_t I) const { return (*this)[I].asArray(); }
    ArrayExpr *getArray(size_t I) { return (*this)[I].asArray(); }
  };

private:
  mutable llvm::AlignedCharArrayUnion<bool, double, llvm::StringRef,
                                      std::string, ArrayExpr, ObjectExpr>
      Union;
};

bool operator==(const Expr &, const Expr &);
inline bool operator!=(const Expr &L, const Expr &R) { return !(L == R); }
inline bool operator==(const Expr::ObjectKey &L, const Expr::ObjectKey &R) {
  return llvm::StringRef(L) == llvm::StringRef(R);
}
inline bool operator!=(const Expr::ObjectKey &L, const Expr::ObjectKey &R) {
  return !(L == R);
}

struct Expr::ObjectExpr::KV {
  ObjectKey K;
  Expr V;
};

inline Expr::ObjectExpr::ObjectExpr(std::initializer_list<KV> Properties) {
  for (const auto &P : Properties)
    emplace(std::move(P.K), std::move(P.V));
}

// Give Expr::{Object,Array} more convenient names for literal use.
using obj = Expr::ObjectExpr;
using ary = Expr::ArrayExpr;

llvm::Expected<Expr> parse(llvm::StringRef JSON);

class ParseError : public llvm::ErrorInfo<ParseError> {
  const char *Msg;
  unsigned Line, Column, Offset;

public:
  static char ID;
  ParseError(const char *Msg, unsigned Line, unsigned Column, unsigned Offset)
      : Msg(Msg), Line(Line), Column(Column), Offset(Offset) {}
  void log(llvm::raw_ostream &OS) const override {
    OS << llvm::formatv("[{0}:{1}, byte={2}]: {3}", Line, Column, Offset, Msg);
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

} // namespace json
} // namespace clangd
} // namespace clang

namespace llvm {
template <> struct format_provider<clang::clangd::json::Expr> {
  static void format(const clang::clangd::json::Expr &, raw_ostream &,
                     StringRef);
};
} // namespace llvm

#endif
