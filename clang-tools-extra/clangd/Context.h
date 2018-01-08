//===--- Context.h - Mechanism for passing implicit data --------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Context for storing and retrieving implicit data. Useful for passing implicit
// parameters on a per-request basis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONTEXT_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONTEXT_H_

#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <type_traits>

namespace clang {
namespace clangd {

/// A key for a value of type \p Type, stored inside a context. Keys are
/// non-movable and non-copyable. See documentation of the Context class for
/// more details and usage examples.
template <class Type> class Key {
public:
  static_assert(!std::is_reference<Type>::value,
                "Reference arguments to Key<> are not allowed");

  Key() = default;

  Key(Key const &) = delete;
  Key &operator=(Key const &) = delete;
  Key(Key &&) = delete;
  Key &operator=(Key &&) = delete;
};

/// A context is an immutable container for per-request data that must be
/// propagated through layers that don't care about it. An example is a request
/// ID that we may want to use when logging.
///
/// Conceptually, a context is a heterogeneous map<Key<T>, T>. Each key has
/// an associated value type, which allows the map to be typesafe.
///
/// You can't add data to an existing context, instead you create a new
/// immutable context derived from it with extra data added. When you retrieve
/// data, the context will walk up the parent chain until the key is found.
///
/// Contexts should be:
///  - passed by reference when calling synchronous functions
///  - passed by value (move) when calling asynchronous functions. The result
///    callback of async operations will receive the context again.
///  - cloned only when 'forking' an asynchronous computation that we don't wait
///    for.
///
/// Copy operations for this class are deleted, use an explicit clone() method
/// when you need a copy of the context instead.
///
/// To derive a child context use derive() function, e.g.
///     Context ChildCtx = ParentCtx.derive(RequestIdKey, 123);
///
/// To create a new root context, derive() from empty Context.
/// e.g.:
///     Context Ctx = Context::empty().derive(RequestIdKey, 123);
///
/// Values in the context are indexed by typed keys (instances of Key<T> class).
/// Key<T> serves two purposes:
///   - it provides a lookup key for the context (each instance of a key is
///   unique),
///   - it keeps the type information about the value stored in the context map
///   in the template arguments.
/// This provides a type-safe interface to store and access values of multiple
/// types inside a single context.
/// For example,
///    Key<int> RequestID;
///    Key<int> Version;
///
///    Context Ctx = Context::empty().derive(RequestID, 10).derive(Version, 3);
///    assert(*Ctx.get(RequestID) == 10);
///    assert(*Ctx.get(Version) == 3);
///
/// Keys are typically used across multiple functions, so most of the time you
/// would want to make them static class members or global variables.
class Context {
public:
  /// Returns an empty context that contains no data. Useful for calling
  /// functions that require a context when no explicit context is available.
  static Context empty();

private:
  struct Data;
  Context(std::shared_ptr<const Data> DataPtr);

public:
  /// Same as Context::empty(), please use Context::empty() instead.
  /// Constructor is defined to workaround a bug in MSVC's version of STL.
  /// (arguments of std::future<> must be default-construcitble in MSVC).
  Context() = default;

  /// Move-only.
  Context(Context const &) = delete;
  Context &operator=(const Context &) = delete;

  Context(Context &&) = default;
  Context &operator=(Context &&) = default;

  /// Get data stored for a typed \p Key. If values are not found
  /// \returns Pointer to the data associated with \p Key. If no data is
  /// specified for \p Key, return null.
  template <class Type> const Type *get(const Key<Type> &Key) const {
    for (const Data *DataPtr = this->DataPtr.get(); DataPtr != nullptr;
         DataPtr = DataPtr->Parent.get()) {
      if (DataPtr->KeyPtr == &Key)
        return static_cast<const Type *>(DataPtr->Value->getValuePtr());
    }
    return nullptr;
  }

  /// A helper to get a reference to a \p Key that must exist in the map.
  /// Must not be called for keys that are not in the map.
  template <class Type> const Type &getExisting(const Key<Type> &Key) const {
    auto Val = get(Key);
    assert(Val && "Key does not exist");
    return *Val;
  }

  /// Derives a child context
  /// It is safe to move or destroy a parent context after calling derive() from
  /// it. The child context will continue to have access to the data stored in
  /// the parent context.
  template <class Type>
  Context derive(const Key<Type> &Key,
                 typename std::decay<Type>::type Value) const & {
    return Context(std::make_shared<Data>(Data{
        /*Parent=*/DataPtr, &Key,
        llvm::make_unique<TypedAnyStorage<typename std::decay<Type>::type>>(
            std::move(Value))}));
  }

  template <class Type>
  Context
  derive(const Key<Type> &Key,
         typename std::decay<Type>::type Value) && /* takes ownership */ {
    return Context(std::make_shared<Data>(Data{
        /*Parent=*/std::move(DataPtr), &Key,
        llvm::make_unique<TypedAnyStorage<typename std::decay<Type>::type>>(
            std::move(Value))}));
  }

  /// Clone this context object.
  Context clone() const;

private:
  class AnyStorage {
  public:
    virtual ~AnyStorage() = default;
    virtual void *getValuePtr() = 0;
  };

  template <class T> class TypedAnyStorage : public Context::AnyStorage {
    static_assert(std::is_same<typename std::decay<T>::type, T>::value,
                  "Argument to TypedAnyStorage must be decayed");

  public:
    TypedAnyStorage(T &&Value) : Value(std::move(Value)) {}

    void *getValuePtr() override { return &Value; }

  private:
    T Value;
  };

  struct Data {
    // We need to make sure Parent outlives the Value, so the order of members
    // is important. We do that to allow classes stored in Context's child
    // layers to store references to the data in the parent layers.
    std::shared_ptr<const Data> Parent;
    const void *KeyPtr;
    std::unique_ptr<AnyStorage> Value;
  };

  std::shared_ptr<const Data> DataPtr;
}; // namespace clangd

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CONTEXT_H_
