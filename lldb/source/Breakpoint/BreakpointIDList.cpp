//===-- BreakpointIDList.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointIDList.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// class BreakpointIDList
//----------------------------------------------------------------------

BreakpointIDList::BreakpointIDList()
    : m_invalid_id(LLDB_INVALID_BREAK_ID, LLDB_INVALID_BREAK_ID) {}

BreakpointIDList::~BreakpointIDList() = default;

size_t BreakpointIDList::GetSize() const { return m_breakpoint_ids.size(); }

const BreakpointID &
BreakpointIDList::GetBreakpointIDAtIndex(size_t index) const {
  return ((index < m_breakpoint_ids.size()) ? m_breakpoint_ids[index]
                                            : m_invalid_id);
}

bool BreakpointIDList::RemoveBreakpointIDAtIndex(size_t index) {
  if (index >= m_breakpoint_ids.size())
    return false;

  m_breakpoint_ids.erase(m_breakpoint_ids.begin() + index);
  return true;
}

void BreakpointIDList::Clear() { m_breakpoint_ids.clear(); }

bool BreakpointIDList::AddBreakpointID(BreakpointID bp_id) {
  m_breakpoint_ids.push_back(bp_id);

  return true; // We don't do any verification in this function, so always
               // return true.
}

bool BreakpointIDList::AddBreakpointID(const char *bp_id_str) {
  BreakpointID temp_bp_id;
  break_id_t bp_id;
  break_id_t loc_id;

  bool success =
      BreakpointID::ParseCanonicalReference(bp_id_str, &bp_id, &loc_id);

  if (success) {
    temp_bp_id.SetID(bp_id, loc_id);
    m_breakpoint_ids.push_back(temp_bp_id);
  }

  return success;
}

bool BreakpointIDList::FindBreakpointID(BreakpointID &bp_id,
                                        size_t *position) const {
  for (size_t i = 0; i < m_breakpoint_ids.size(); ++i) {
    BreakpointID tmp_id = m_breakpoint_ids[i];
    if (tmp_id.GetBreakpointID() == bp_id.GetBreakpointID() &&
        tmp_id.GetLocationID() == bp_id.GetLocationID()) {
      *position = i;
      return true;
    }
  }

  return false;
}

bool BreakpointIDList::FindBreakpointID(const char *bp_id_str,
                                        size_t *position) const {
  BreakpointID temp_bp_id;
  break_id_t bp_id;
  break_id_t loc_id;

  if (BreakpointID::ParseCanonicalReference(bp_id_str, &bp_id, &loc_id)) {
    temp_bp_id.SetID(bp_id, loc_id);
    return FindBreakpointID(temp_bp_id, position);
  } else
    return false;
}

void BreakpointIDList::InsertStringArray(const char **string_array,
                                         size_t array_size,
                                         CommandReturnObject &result) {
  if (string_array == nullptr)
    return;

  for (uint32_t i = 0; i < array_size; ++i) {
    break_id_t bp_id;
    break_id_t loc_id;

    if (BreakpointID::ParseCanonicalReference(string_array[i], &bp_id,
                                              &loc_id)) {
      if (bp_id != LLDB_INVALID_BREAK_ID) {
        BreakpointID temp_bp_id(bp_id, loc_id);
        m_breakpoint_ids.push_back(temp_bp_id);
      } else {
        result.AppendErrorWithFormat("'%s' is not a valid breakpoint ID.\n",
                                     string_array[i]);
        result.SetStatus(eReturnStatusFailed);
        return;
      }
    }
  }
  result.SetStatus(eReturnStatusSuccessFinishNoResult);
}

//  This function takes OLD_ARGS, which is usually the result of breaking the
//  command string arguments into
//  an array of space-separated strings, and searches through the arguments for
//  any breakpoint ID range specifiers.
//  Any string in the array that is not part of an ID range specifier is copied
//  directly into NEW_ARGS.  If any
//  ID range specifiers are found, the range is interpreted and a list of
//  canonical breakpoint IDs corresponding to
//  all the current breakpoints and locations in the range are added to
//  NEW_ARGS.  When this function is done,
//  NEW_ARGS should be a copy of OLD_ARGS, with and ID range specifiers replaced
//  by the members of the range.

void BreakpointIDList::FindAndReplaceIDRanges(Args &old_args, Target *target,
                                              bool allow_locations,
                                              CommandReturnObject &result,
                                              Args &new_args) {
  std::string range_start;
  const char *range_end;
  const char *current_arg;
  const size_t num_old_args = old_args.GetArgumentCount();
  std::set<std::string> names_found;

  for (size_t i = 0; i < num_old_args; ++i) {
    bool is_range = false;

    current_arg = old_args.GetArgumentAtIndex(i);
    if (!allow_locations && strchr(current_arg, '.') != nullptr) {
      result.AppendErrorWithFormat(
          "Breakpoint locations not allowed, saw location: %s.", current_arg);
      new_args.Clear();
      return;
    }

    size_t range_start_len = 0;
    size_t range_end_pos = 0;
    Error error;

    if (BreakpointIDList::StringContainsIDRangeExpression(
            current_arg, &range_start_len, &range_end_pos)) {
      is_range = true;
      range_start.assign(current_arg, range_start_len);
      range_end = current_arg + range_end_pos;
    } else if (BreakpointID::StringIsBreakpointName(
                   llvm::StringRef(current_arg), error)) {
      if (!error.Success()) {
        new_args.Clear();
        result.AppendError(error.AsCString());
        result.SetStatus(eReturnStatusFailed);
        return;
      } else
        names_found.insert(current_arg);
    } else if ((i + 2 < num_old_args) &&
               BreakpointID::IsRangeIdentifier(
                   old_args.GetArgumentAtIndex(i + 1)) &&
               BreakpointID::IsValidIDExpression(current_arg) &&
               BreakpointID::IsValidIDExpression(
                   old_args.GetArgumentAtIndex(i + 2))) {
      range_start.assign(current_arg);
      range_end = old_args.GetArgumentAtIndex(i + 2);
      is_range = true;
      i = i + 2;
    } else {
      // See if user has specified id.*
      std::string tmp_str = old_args.GetArgumentAtIndex(i);
      size_t pos = tmp_str.find('.');
      if (pos != std::string::npos) {
        std::string bp_id_str = tmp_str.substr(0, pos);
        if (BreakpointID::IsValidIDExpression(bp_id_str.c_str()) &&
            tmp_str[pos + 1] == '*' && tmp_str.length() == (pos + 2)) {
          break_id_t bp_id;
          break_id_t bp_loc_id;

          BreakpointID::ParseCanonicalReference(bp_id_str.c_str(), &bp_id,
                                                &bp_loc_id);
          BreakpointSP breakpoint_sp = target->GetBreakpointByID(bp_id);
          if (!breakpoint_sp) {
            new_args.Clear();
            result.AppendErrorWithFormat("'%d' is not a valid breakpoint ID.\n",
                                         bp_id);
            result.SetStatus(eReturnStatusFailed);
            return;
          }
          const size_t num_locations = breakpoint_sp->GetNumLocations();
          for (size_t j = 0; j < num_locations; ++j) {
            BreakpointLocation *bp_loc =
                breakpoint_sp->GetLocationAtIndex(j).get();
            StreamString canonical_id_str;
            BreakpointID::GetCanonicalReference(&canonical_id_str, bp_id,
                                                bp_loc->GetID());
            new_args.AppendArgument(canonical_id_str.GetData());
          }
        }
      }
    }

    if (is_range) {
      break_id_t start_bp_id;
      break_id_t end_bp_id;
      break_id_t start_loc_id;
      break_id_t end_loc_id;

      BreakpointID::ParseCanonicalReference(range_start.c_str(), &start_bp_id,
                                            &start_loc_id);
      BreakpointID::ParseCanonicalReference(range_end, &end_bp_id, &end_loc_id);

      if ((start_bp_id == LLDB_INVALID_BREAK_ID) ||
          (!target->GetBreakpointByID(start_bp_id))) {
        new_args.Clear();
        result.AppendErrorWithFormat("'%s' is not a valid breakpoint ID.\n",
                                     range_start.c_str());
        result.SetStatus(eReturnStatusFailed);
        return;
      }

      if ((end_bp_id == LLDB_INVALID_BREAK_ID) ||
          (!target->GetBreakpointByID(end_bp_id))) {
        new_args.Clear();
        result.AppendErrorWithFormat("'%s' is not a valid breakpoint ID.\n",
                                     range_end);
        result.SetStatus(eReturnStatusFailed);
        return;
      }

      if (((start_loc_id == LLDB_INVALID_BREAK_ID) &&
           (end_loc_id != LLDB_INVALID_BREAK_ID)) ||
          ((start_loc_id != LLDB_INVALID_BREAK_ID) &&
           (end_loc_id == LLDB_INVALID_BREAK_ID))) {
        new_args.Clear();
        result.AppendErrorWithFormat("Invalid breakpoint id range:  Either "
                                     "both ends of range must specify"
                                     " a breakpoint location, or neither can "
                                     "specify a breakpoint location.\n");
        result.SetStatus(eReturnStatusFailed);
        return;
      }

      // We have valid range starting & ending breakpoint IDs.  Go through all
      // the breakpoints in the
      // target and find all the breakpoints that fit into this range, and add
      // them to new_args.

      // Next check to see if we have location id's.  If so, make sure the
      // start_bp_id and end_bp_id are
      // for the same breakpoint; otherwise we have an illegal range: breakpoint
      // id ranges that specify
      // bp locations are NOT allowed to cross major bp id numbers.

      if ((start_loc_id != LLDB_INVALID_BREAK_ID) ||
          (end_loc_id != LLDB_INVALID_BREAK_ID)) {
        if (start_bp_id != end_bp_id) {
          new_args.Clear();
          result.AppendErrorWithFormat(
              "Invalid range: Ranges that specify particular breakpoint "
              "locations"
              " must be within the same major breakpoint; you specified two"
              " different major breakpoints, %d and %d.\n",
              start_bp_id, end_bp_id);
          result.SetStatus(eReturnStatusFailed);
          return;
        }
      }

      const BreakpointList &breakpoints = target->GetBreakpointList();
      const size_t num_breakpoints = breakpoints.GetSize();
      for (size_t j = 0; j < num_breakpoints; ++j) {
        Breakpoint *breakpoint = breakpoints.GetBreakpointAtIndex(j).get();
        break_id_t cur_bp_id = breakpoint->GetID();

        if ((cur_bp_id < start_bp_id) || (cur_bp_id > end_bp_id))
          continue;

        const size_t num_locations = breakpoint->GetNumLocations();

        if ((cur_bp_id == start_bp_id) &&
            (start_loc_id != LLDB_INVALID_BREAK_ID)) {
          for (size_t k = 0; k < num_locations; ++k) {
            BreakpointLocation *bp_loc =
                breakpoint->GetLocationAtIndex(k).get();
            if ((bp_loc->GetID() >= start_loc_id) &&
                (bp_loc->GetID() <= end_loc_id)) {
              StreamString canonical_id_str;
              BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                  bp_loc->GetID());
              new_args.AppendArgument(canonical_id_str.GetData());
            }
          }
        } else if ((cur_bp_id == end_bp_id) &&
                   (end_loc_id != LLDB_INVALID_BREAK_ID)) {
          for (size_t k = 0; k < num_locations; ++k) {
            BreakpointLocation *bp_loc =
                breakpoint->GetLocationAtIndex(k).get();
            if (bp_loc->GetID() <= end_loc_id) {
              StreamString canonical_id_str;
              BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                  bp_loc->GetID());
              new_args.AppendArgument(canonical_id_str.GetData());
            }
          }
        } else {
          StreamString canonical_id_str;
          BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                              LLDB_INVALID_BREAK_ID);
          new_args.AppendArgument(canonical_id_str.GetData());
        }
      }
    } else // else is_range was false
    {
      new_args.AppendArgument(current_arg);
    }
  }

  // Okay, now see if we found any names, and if we did, add them:
  if (target && names_found.size()) {
    for (BreakpointSP bkpt_sp : target->GetBreakpointList().Breakpoints()) {
      for (std::string name : names_found) {
        if (bkpt_sp->MatchesName(name.c_str())) {
          StreamString canonical_id_str;
          BreakpointID::GetCanonicalReference(
              &canonical_id_str, bkpt_sp->GetID(), LLDB_INVALID_BREAK_ID);
          new_args.AppendArgument(canonical_id_str.GetData());
        }
      }
    }
  }

  result.SetStatus(eReturnStatusSuccessFinishNoResult);
}

bool BreakpointIDList::StringContainsIDRangeExpression(const char *in_string,
                                                       size_t *range_start_len,
                                                       size_t *range_end_pos) {
  bool is_range_expression = false;
  std::string arg_str = in_string;
  std::string::size_type idx;
  std::string::size_type start_pos = 0;

  *range_start_len = 0;
  *range_end_pos = 0;

  int specifiers_size = 0;
  for (int i = 0; BreakpointID::g_range_specifiers[i] != nullptr; ++i)
    ++specifiers_size;

  for (int i = 0; i < specifiers_size && !is_range_expression; ++i) {
    const char *specifier_str = BreakpointID::g_range_specifiers[i];
    size_t len = strlen(specifier_str);
    idx = arg_str.find(BreakpointID::g_range_specifiers[i]);
    if (idx != std::string::npos) {
      *range_start_len = idx - start_pos;
      std::string start_str = arg_str.substr(start_pos, *range_start_len);
      if (idx + len < arg_str.length()) {
        *range_end_pos = idx + len;
        std::string end_str = arg_str.substr(*range_end_pos);
        if (BreakpointID::IsValidIDExpression(start_str.c_str()) &&
            BreakpointID::IsValidIDExpression(end_str.c_str())) {
          is_range_expression = true;
          //*range_start = start_str;
          //*range_end = end_str;
        }
      }
    }
  }

  if (!is_range_expression) {
    *range_start_len = 0;
    *range_end_pos = 0;
  }

  return is_range_expression;
}
