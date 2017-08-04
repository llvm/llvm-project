//===-- VMRange.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"

#include "lldb/Core/Stream.h"
#include "lldb/Core/VMRange.h"
#include <algorithm>

using namespace lldb;
using namespace lldb_private;

bool VMRange::ContainsValue(const VMRange::collection &coll,
                            lldb::addr_t value) {
  ValueInRangeUnaryPredicate in_range_predicate(value);
  VMRange::const_iterator pos;
  VMRange::const_iterator end = coll.end();
  pos = std::find_if(coll.begin(), end, in_range_predicate);
  if (pos != end)
    return true;
  return false;
}

bool VMRange::ContainsRange(const VMRange::collection &coll,
                            const VMRange &range) {
  RangeInRangeUnaryPredicate in_range_predicate(range);
  VMRange::const_iterator pos;
  VMRange::const_iterator end = coll.end();
  pos = std::find_if(coll.begin(), end, in_range_predicate);
  if (pos != end)
    return true;
  return false;
}

size_t VMRange::FindRangeIndexThatContainsValue(const VMRange::collection &coll,
                                                lldb::addr_t value) {
  ValueInRangeUnaryPredicate in_range_predicate(value);
  VMRange::const_iterator begin = coll.begin();
  VMRange::const_iterator end = coll.end();
  VMRange::const_iterator pos = std::find_if(begin, end, in_range_predicate);
  if (pos != end)
    return std::distance(begin, pos);
  return UINT32_MAX;
}

void VMRange::Dump(Stream *s, lldb::addr_t offset, uint32_t addr_width) const {
  s->AddressRange(offset + GetBaseAddress(), offset + GetEndAddress(),
                  addr_width);
}

bool lldb_private::operator==(const VMRange &lhs, const VMRange &rhs) {
  return lhs.GetBaseAddress() == rhs.GetBaseAddress() &&
         lhs.GetEndAddress() == rhs.GetEndAddress();
}

bool lldb_private::operator!=(const VMRange &lhs, const VMRange &rhs) {
  return lhs.GetBaseAddress() != rhs.GetBaseAddress() ||
         lhs.GetEndAddress() != rhs.GetEndAddress();
}

bool lldb_private::operator<(const VMRange &lhs, const VMRange &rhs) {
  if (lhs.GetBaseAddress() < rhs.GetBaseAddress())
    return true;
  else if (lhs.GetBaseAddress() > rhs.GetBaseAddress())
    return false;
  return lhs.GetEndAddress() < rhs.GetEndAddress();
}

bool lldb_private::operator<=(const VMRange &lhs, const VMRange &rhs) {
  if (lhs.GetBaseAddress() < rhs.GetBaseAddress())
    return true;
  else if (lhs.GetBaseAddress() > rhs.GetBaseAddress())
    return false;
  return lhs.GetEndAddress() <= rhs.GetEndAddress();
}

bool lldb_private::operator>(const VMRange &lhs, const VMRange &rhs) {
  if (lhs.GetBaseAddress() > rhs.GetBaseAddress())
    return true;
  else if (lhs.GetBaseAddress() < rhs.GetBaseAddress())
    return false;
  return lhs.GetEndAddress() > rhs.GetEndAddress();
}

bool lldb_private::operator>=(const VMRange &lhs, const VMRange &rhs) {
  if (lhs.GetBaseAddress() > rhs.GetBaseAddress())
    return true;
  else if (lhs.GetBaseAddress() < rhs.GetBaseAddress())
    return false;
  return lhs.GetEndAddress() >= rhs.GetEndAddress();
}
