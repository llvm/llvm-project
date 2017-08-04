//===--------------------- Range.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_Range_h_
#define utility_Range_h_

#include <algorithm>
#include <stdint.h>

namespace lldb_utility {

class Range {
public:
  typedef uint64_t ValueType;

  static const ValueType OPEN_END = UINT64_MAX;

  Range(const Range &rng);

  Range(ValueType low = 0, ValueType high = OPEN_END);

  Range &operator=(const Range &rhs);

  ValueType GetLow() { return m_low; }

  ValueType GetHigh() { return m_high; }

  void SetLow(ValueType low) { m_low = low; }

  void SetHigh(ValueType high) { m_high = high; }

  void Flip();

  void Intersection(const Range &other);

  void Union(const Range &other);

  typedef bool (*RangeCallback)(ValueType index);

  void Iterate(RangeCallback callback);

  ValueType GetSize();

  bool IsEmpty();

private:
  void InitRange();

  ValueType m_low;
  ValueType m_high;
};

} // namespace lldb_private

#endif // #ifndef utility_Range_h_
