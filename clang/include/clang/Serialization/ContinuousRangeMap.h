//===--- ContinuousRangeMap.h - Map with int range as key -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ContinuousRangeMap class, which is a highly
//  specialized container used by serialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_CONTINUOUS_RANGE_MAP_H
#define LLVM_CLANG_SERIALIZATION_CONTINUOUS_RANGE_MAP_H

#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <utility>

namespace clang {

/// \brief A map from continuous integer ranges to some value, with a very
/// specialized interface.
///
/// CRM maps from integer ranges to values. The ranges are continuous, i.e.
/// where one ends, the next one begins. So if the map contains the stops I0-3,
/// the first range is from I0 to I1, the second from I1 to I2, the third from
/// I2 to I3 and the last from I3 to infinity.
///
/// Ranges must be inserted in order. Inserting a new stop I4 into the map will
/// shrink the fourth range to I3 to I4 and add the new range I4 to inf.
template <typename Int, typename V, unsigned InitialCapacity>
class ContinuousRangeMap {
public:
  typedef std::pair<Int, V> value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;

private:
  typedef SmallVector<value_type, InitialCapacity> Representation;
  Representation Rep;

  struct Compare {
    bool operator ()(const_reference L, Int R) const {
      return L.first < R;
    }
    bool operator ()(Int L, const_reference R) const {
      return L < R.first;
    }
    bool operator ()(Int L, Int R) const { 
      return L < R;
    }
    bool operator ()(const_reference L, const_reference R) const {
      return L.first < R.first;
    }
  };

public:
  void insert(const value_type &Val) {
    if (!Rep.empty() && Rep.back() == Val)
      return;

    assert((Rep.empty() || Rep.back().first < Val.first) &&
           "Must insert keys in order.");
    Rep.push_back(Val);
  }
  
  void insertOrReplace(const value_type &Val) {
    iterator I = std::lower_bound(Rep.begin(), Rep.end(), Val, Compare());
    if (I != Rep.end() && I->first == Val.first) {
      I->second = Val.second;
      return;
    }
    
    Rep.insert(I, Val);
  }

  typedef typename Representation::iterator iterator;
  typedef typename Representation::const_iterator const_iterator;

  iterator begin() { return Rep.begin(); }
  iterator end() { return Rep.end(); }
  const_iterator begin() const { return Rep.begin(); }
  const_iterator end() const { return Rep.end(); }

  iterator find(Int K) {
    iterator I = std::upper_bound(Rep.begin(), Rep.end(), K, Compare());
    // I points to the first entry with a key > K, which is the range that
    // follows the one containing K.
    if (I == Rep.begin())
      return Rep.end();
    --I;
    return I;
  }
  const_iterator find(Int K) const {
    return const_cast<ContinuousRangeMap*>(this)->find(K);
  }

  reference back() { return Rep.back(); }
  const_reference back() const { return Rep.back(); }
  
  /// \brief An object that helps properly build a continuous range map
  /// from a set of values.
  class Builder {
    ContinuousRangeMap &Self;
    
    Builder(const Builder&) LLVM_DELETED_FUNCTION;
    Builder &operator=(const Builder&) LLVM_DELETED_FUNCTION;
    
  public:
    explicit Builder(ContinuousRangeMap &Self) : Self(Self) { }
    
    ~Builder() {
      std::sort(Self.Rep.begin(), Self.Rep.end(), Compare());
    }
    
    void insert(const value_type &Val) {
      Self.Rep.push_back(Val);
    }
  };
  friend class Builder;
};

}

#endif
