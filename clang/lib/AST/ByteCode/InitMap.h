//===----------------------- InitMap.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INIT_MAP_H
#define LLVM_CLANG_AST_INTERP_INIT_MAP_H

#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>

namespace clang {
namespace interp {

/// Bitfield tracking the initialisation status of elements of primitive arrays.
struct InitMap final {
private:
  /// Type packing bits.
  using T = uint64_t;
  /// Bits stored in a single field.
  static constexpr uint64_t PER_FIELD = sizeof(T) * CHAR_BIT;
  /// Number of fields in the init map.
  unsigned NumElems;
  /// Number of fields not initialized.
  unsigned UninitFields;
  unsigned DeadFields = 0;
  std::unique_ptr<T[]> Data;

public:
  /// Initializes the map with no fields set.
  explicit InitMap(unsigned N)
      : NumElems(N), UninitFields(N),
        Data(std::make_unique<T[]>(numFields(N))) {}
  explicit InitMap(unsigned N, bool AllInitialized)
      : NumElems(N), UninitFields(AllInitialized ? 0 : N),
        Data(std::make_unique<T[]>(numFields(N))) {
    if (AllInitialized) {
      for (unsigned I = 0; I != (numFields(N) / 2); ++I)
        Data[I] = std::numeric_limits<T>::max();
    }
  }

  void startElementLifetime(unsigned I);
  void endElementLifetime(unsigned I);

  bool isElementAlive(unsigned I) const {
    unsigned LifetimeIndex = (NumElems + I);
    unsigned Bucket = numFields(NumElems) / 2 + (I / PER_FIELD);
    return !(data()[Bucket] & (T(1) << (LifetimeIndex % PER_FIELD)));
  }

  bool allElementsAlive() const { return DeadFields == 0; }

  /// Initializes an element. Returns true when object if fully initialized.
  bool initializeElement(unsigned I);

  /// Checks if an element was initialized.
  bool isElementInitialized(unsigned I) const;

private:
  /// Returns a pointer to storage.
  T *data() { return Data.get(); }
  const T *data() const { return Data.get(); }

  static constexpr size_t numFields(unsigned N) {
    return ((N + PER_FIELD - 1) / PER_FIELD) * 2;
  }
};

/// A pointer-sized struct we use to allocate into data storage.
/// An InitMapPtr is either backed by an actual InitMap, or it
/// hold information about the absence of the InitMap.
struct InitMapPtr final {
  /// V's value before an initmap has been created.
  static constexpr intptr_t NoInitMapValue = 0;
  /// V's value after the initmap has been destroyed because
  /// all its elements have already been initialized.
  static constexpr intptr_t AllInitializedValue = 1;
  uintptr_t V = 0;

  explicit InitMapPtr() = default;
  bool hasInitMap() const {
    return V != NoInitMapValue && V != AllInitializedValue;
  }
  /// Are all elements in the array already initialized?
  bool allInitialized() const { return V == AllInitializedValue; }

  void setInitMap(const InitMap *IM) {
    assert(IM != nullptr);
    V = reinterpret_cast<uintptr_t>(IM);
    assert(hasInitMap());
  }

  void noteAllInitialized() {
    if (hasInitMap())
      delete (operator->)();
    V = AllInitializedValue;
  }

  /// Access the underlying InitMap directly.
  InitMap *operator->() {
    assert(hasInitMap());
    return reinterpret_cast<InitMap *>(V);
  }

  /// Delete the InitMap if one exists.
  void deleteInitMap() {
    if (hasInitMap())
      delete (operator->)();
    V = NoInitMapValue;
  };
};
static_assert(sizeof(InitMapPtr) == sizeof(void *));
} // namespace interp
} // namespace clang

#endif
