//===-- BreakpointIDList.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointIDList_h_
#define liblldb_BreakpointIDList_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
// class BreakpointIDList
//----------------------------------------------------------------------

class BreakpointIDList {
public:
  typedef std::vector<BreakpointID> BreakpointIDArray;

  BreakpointIDList();

  virtual ~BreakpointIDList();

  size_t GetSize() const;

  const BreakpointID &GetBreakpointIDAtIndex(size_t index) const;

  bool RemoveBreakpointIDAtIndex(size_t index);

  void Clear();

  bool AddBreakpointID(BreakpointID bp_id);

  bool AddBreakpointID(const char *bp_id);

  bool FindBreakpointID(BreakpointID &bp_id, size_t *position) const;

  bool FindBreakpointID(const char *bp_id, size_t *position) const;

  void InsertStringArray(const char **string_array, size_t array_size,
                         CommandReturnObject &result);

  static bool StringContainsIDRangeExpression(const char *in_string,
                                              size_t *range_start_len,
                                              size_t *range_end_pos);

  static void FindAndReplaceIDRanges(Args &old_args, Target *target,
                                     bool allow_locations,
                                     CommandReturnObject &result,
                                     Args &new_args);

private:
  BreakpointIDArray m_breakpoint_ids;
  BreakpointID m_invalid_id;

  DISALLOW_COPY_AND_ASSIGN(BreakpointIDList);
};

} // namespace lldb_private

#endif // liblldb_BreakpointIDList_h_
