//===--- JSONExpr.h - composable JSON expressions ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSON_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_JSON_H

#include <map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace json {

// An Expr is an opaque temporary JSON structure used to compose documents.
// They can be copied, but should generally be moved.
//
// You can implicitly construct literals from:
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
// Exprs can be serialized to JSON:
//   1) raw_ostream << Expr                    // Basic formatting.
//   2) raw_ostream << formatv("{0}", Expr)    // Basic formatting.
//   3) raw_ostream << formatv("{0:2}", Expr)  // Pretty-print with indent 2.
class Expr {
public:
  class Object;
  class ObjectKey;
  class Array;

  // It would be nice to have Expr() be null. But that would make {} null too...
  Expr(const Expr &M) { copyFrom(M); }
  Expr(Expr &&M) { moveFrom(std::move(M)); }
  // "cheating" move-constructor for moving from initializer_list.
  Expr(const Expr &&M) { moveFrom(std::move(M)); }
  Expr(std::initializer_list<Expr> Elements) : Expr(Array(Elements)) {}
  Expr(Array &&Elements) : Type(T_Array) { create<Array>(std::move(Elements)); }
  Expr(Object &&Properties) : Type(T_Object) {
    create<Object>(std::move(Properties));
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
  // ObjectKey is a used to capture keys in Expr::Objects. It's like Expr but:
  //   - only strings are allowed
  //   - it's copyable (for std::map)
  //   - we're slightly more eager to copy, to allow efficient key compares
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

  class Object : public std::map<ObjectKey, Expr> {
  public:
    explicit Object() {}
    // Use a custom struct for list-init, because pair forces extra copies.
    struct KV;
    explicit Object(std::initializer_list<KV> Properties);

    // Allow [] as if Expr was default-constructible as null.
    Expr &operator[](const ObjectKey &K) {
      return emplace(K, Expr(nullptr)).first->second;
    }
    Expr &operator[](ObjectKey &&K) {
      return emplace(std::move(K), Expr(nullptr)).first->second;
    }
  };

  class Array : public std::vector<Expr> {
  public:
    explicit Array() {}
    explicit Array(std::initializer_list<Expr> Elements) {
      reserve(Elements.size());
      for (const Expr &V : Elements)
        emplace_back(std::move(V));
    };
    template <typename Collection> explicit Array(const Collection &C) {
      for (const auto &V : C)
        emplace_back(V);
    }
  };

private:
  mutable llvm::AlignedCharArrayUnion<bool, double, llvm::StringRef,
                                      std::string, Array, Object>
      Union;
};

struct Expr::Object::KV {
  ObjectKey K;
  Expr V;
};

inline Expr::Object::Object(std::initializer_list<KV> Properties) {
  for (const auto &P : Properties)
    emplace(std::move(P.K), std::move(P.V));
}

// Give Expr::{Object,Array} more convenient names for literal use.
using obj = Expr::Object;
using ary = Expr::Array;

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
