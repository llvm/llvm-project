//ProgramStateTrait.h - Partial implementations of ProgramStateTrait -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines partial implementations of template specializations of
//  the class ProgramStateTrait<>.  ProgramStateTrait<> is used by ProgramState
//  to implement set/get methods for manipulating a ProgramState's
//  generic data map.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATETRAIT_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATETRAIT_H

#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/Allocator.h"
#include <cstdint>

namespace clang {
namespace ento {

  template <typename T> struct ProgramStatePartialTrait;

  /// Declares a program state trait for type \p Type called \p Name, and
  /// introduce a type named \c NameTy.
  /// The macro should not be used inside namespaces, or for traits that must
  /// be accessible from more than one translation unit.
  #define REGISTER_TRAIT_WITH_PROGRAMSTATE(Name, Type) \
    namespace { \
      class Name {}; \
      using Name ## Ty = Type; \
    } \
    namespace clang { \
    namespace ento { \
      template <> \
      struct ProgramStateTrait<Name> \
        : public ProgramStatePartialTrait<Name ## Ty> { \
        static void *GDMIndex() { static int Index; return &Index; } \
      }; \
    } \
    }

  // Partial-specialization for ImmutableMap.
  template <typename Key, typename Data, typename Info>
  struct ProgramStatePartialTrait<llvm::ImmutableMap<Key, Data, Info>> {
    using data_type = llvm::ImmutableMap<Key, Data, Info>;
    using context_type = typename data_type::Factory &;
    using key_type = Key;
    using value_type = Data;
    using lookup_type = const value_type *;

    static data_type MakeData(void *const *p) {
      return p ? data_type((typename data_type::TreeTy *) *p)
               : data_type(nullptr);
    }

    static void *MakeVoidPtr(data_type B) {
      return B.getRoot();
    }

    static lookup_type Lookup(data_type B, key_type K) {
      return B.lookup(K);
    }

    static data_type Set(data_type B, key_type K, value_type E,
                         context_type F) {
      return F.add(B, K, E);
    }

    static data_type Remove(data_type B, key_type K, context_type F) {
      return F.remove(B, K);
    }

    static bool Contains(data_type B, key_type K) {
      return B.contains(K);
    }

    static context_type MakeContext(void *p) {
      return *((typename data_type::Factory *) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator& Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory *) Ctx;
    }
  };

  /// Helper for registering a map trait.
  ///
  /// If the map type were written directly in the invocation of
  /// REGISTER_TRAIT_WITH_PROGRAMSTATE, the comma in the template arguments
  /// would be treated as a macro argument separator, which is wrong.
  /// This allows the user to specify a map type in a way that the preprocessor
  /// can deal with.
  #define CLANG_ENTO_PROGRAMSTATE_MAP(Key, Value) llvm::ImmutableMap<Key, Value>

  // Partial-specialization for ImmutableSet.
  template <typename Key, typename Info>
  struct ProgramStatePartialTrait<llvm::ImmutableSet<Key, Info>> {
    using data_type = llvm::ImmutableSet<Key, Info>;
    using context_type = typename data_type::Factory &;
    using key_type = Key;

    static data_type MakeData(void *const *p) {
      return p ? data_type((typename data_type::TreeTy *) *p)
               : data_type(nullptr);
    }

    static void *MakeVoidPtr(data_type B) {
      return B.getRoot();
    }

    static data_type Add(data_type B, key_type K, context_type F) {
      return F.add(B, K);
    }

    static data_type Remove(data_type B, key_type K, context_type F) {
      return F.remove(B, K);
    }

    static bool Contains(data_type B, key_type K) {
      return B.contains(K);
    }

    static context_type MakeContext(void *p) {
      return *((typename data_type::Factory *) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator &Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory *) Ctx;
    }
  };

  // Partial-specialization for ImmutableList.
  template <typename T>
  struct ProgramStatePartialTrait<llvm::ImmutableList<T>> {
    using data_type = llvm::ImmutableList<T>;
    using key_type = T;
    using context_type = typename data_type::Factory &;

    static data_type Add(data_type L, key_type K, context_type F) {
      return F.add(K, L);
    }

    static bool Contains(data_type L, key_type K) {
      return L.contains(K);
    }

    static data_type MakeData(void *const *p) {
      return p ? data_type((const llvm::ImmutableListImpl<T> *) *p)
               : data_type(nullptr);
    }

    static void *MakeVoidPtr(data_type D) {
      return const_cast<llvm::ImmutableListImpl<T> *>(D.getInternalPointer());
    }

    static context_type MakeContext(void *p) {
      return *((typename data_type::Factory *) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator &Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory *) Ctx;
    }
  };

  // Partial specialization for bool.
  template <> struct ProgramStatePartialTrait<bool> {
    using data_type = bool;

    static data_type MakeData(void *const *p) {
      return p ? (data_type) (uintptr_t) *p
               : data_type();
    }

    static void *MakeVoidPtr(data_type d) {
      return (void *) (uintptr_t) d;
    }
  };

  // Partial specialization for unsigned.
  template <> struct ProgramStatePartialTrait<unsigned> {
    using data_type = unsigned;

    static data_type MakeData(void *const *p) {
      return p ? (data_type) (uintptr_t) *p
               : data_type();
    }

    static void *MakeVoidPtr(data_type d) {
      return (void *) (uintptr_t) d;
    }
  };

  // Partial specialization for void*.
  template <> struct ProgramStatePartialTrait<void *> {
    using data_type = void *;

    static data_type MakeData(void *const *p) {
      return p ? *p
               : data_type();
    }

    static void *MakeVoidPtr(data_type d) {
      return d;
    }
  };

  // Partial specialization for const void *.
  template <> struct ProgramStatePartialTrait<const void *> {
    using data_type = const void *;

    static data_type MakeData(void *const *p) {
      return p ? *p : data_type();
    }

    static void *MakeVoidPtr(data_type d) {
      return const_cast<void *>(d);
    }
  };

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATETRAIT_H
