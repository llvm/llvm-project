//===-- BreakpointID.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointID_h_
#define liblldb_BreakpointID_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
// class BreakpointID
//----------------------------------------------------------------------

class BreakpointID {
public:
  BreakpointID(lldb::break_id_t bp_id = LLDB_INVALID_BREAK_ID,
               lldb::break_id_t loc_id = LLDB_INVALID_BREAK_ID);

  virtual ~BreakpointID();

  lldb::break_id_t GetBreakpointID() const { return m_break_id; }

  lldb::break_id_t GetLocationID() const { return m_location_id; }

  void SetID(lldb::break_id_t bp_id, lldb::break_id_t loc_id) {
    m_break_id = bp_id;
    m_location_id = loc_id;
  }

  void SetBreakpointID(lldb::break_id_t bp_id) { m_break_id = bp_id; }

  void SetBreakpointLocationID(lldb::break_id_t loc_id) {
    m_location_id = loc_id;
  }

  void GetDescription(Stream *s, lldb::DescriptionLevel level);

  static bool IsRangeIdentifier(const char *str);

  static bool IsValidIDExpression(const char *str);

  static const char *g_range_specifiers[];

  //------------------------------------------------------------------
  /// Takes an input string containing the description of a breakpoint or
  /// breakpoint and location
  /// and returns the breakpoint ID and the breakpoint location id.
  ///
  /// @param[in] input
  ///     A string containing JUST the breakpoint description.
  /// @param[out] break_id
  ///     This is the break id.
  /// @param[out] break_loc_id
  ///     This is breakpoint location id, or LLDB_INVALID_BREAK_ID is no
  ///     location was specified.
  /// @return
  ///     \b true if the call was able to extract a breakpoint location from the
  ///     string.  \b false otherwise.
  //------------------------------------------------------------------
  static bool ParseCanonicalReference(const char *input,
                                      lldb::break_id_t *break_id,
                                      lldb::break_id_t *break_loc_id);

  //------------------------------------------------------------------
  /// Takes an input string and checks to see whether it is a breakpoint name.
  /// If it is a mal-formed breakpoint name, error will be set to an appropriate
  /// error string.
  ///
  /// @param[in] input
  ///     A string containing JUST the breakpoint description.
  /// @param[out] error
  ///     If the name is a well-formed breakpoint name, set to success,
  ///     otherwise set to an error.
  /// @return
  ///     \b true if the name is a breakpoint name (as opposed to an ID or
  ///     range) false otherwise.
  //------------------------------------------------------------------
  static bool StringIsBreakpointName(const char *name, Error &error);

  //------------------------------------------------------------------
  /// Takes a breakpoint ID and the breakpoint location id and returns
  /// a string containing the canonical description for the breakpoint
  /// or breakpoint location.
  ///
  /// @param[out] break_id
  ///     This is the break id.
  ///
  /// @param[out] break_loc_id
  ///     This is breakpoint location id, or LLDB_INVALID_BREAK_ID is no
  ///     location is to be specified.
  //------------------------------------------------------------------
  static void GetCanonicalReference(Stream *s, lldb::break_id_t break_id,
                                    lldb::break_id_t break_loc_id);

protected:
  lldb::break_id_t m_break_id;
  lldb::break_id_t m_location_id;
};

} // namespace lldb_private

#endif // liblldb_BreakpointID_h_
