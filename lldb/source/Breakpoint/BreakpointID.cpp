//===-- BreakpointID.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;

BreakpointID::BreakpointID(break_id_t bp_id, break_id_t loc_id)
    : m_break_id(bp_id), m_location_id(loc_id) {}

BreakpointID::~BreakpointID() = default;

const char *BreakpointID::g_range_specifiers[] = {"-", "to", "To", "TO",
                                                  nullptr};

// Tells whether or not STR is valid to use between two strings representing
// breakpoint IDs, to
// indicate a range of breakpoint IDs.  This is broken out into a separate
// function so that we can
// easily change or add to the format for specifying ID ranges at a later date.

bool BreakpointID::IsRangeIdentifier(const char *str) {
  int specifier_count = 0;
  for (int i = 0; g_range_specifiers[i] != nullptr; ++i)
    ++specifier_count;

  for (int i = 0; i < specifier_count; ++i) {
    if (strcmp(g_range_specifiers[i], str) == 0)
      return true;
  }

  return false;
}

bool BreakpointID::IsValidIDExpression(const char *str) {
  break_id_t bp_id;
  break_id_t loc_id;
  BreakpointID::ParseCanonicalReference(str, &bp_id, &loc_id);

  return (bp_id != LLDB_INVALID_BREAK_ID);
}

void BreakpointID::GetDescription(Stream *s, lldb::DescriptionLevel level) {
  if (level == eDescriptionLevelVerbose)
    s->Printf("%p BreakpointID:", static_cast<void *>(this));

  if (m_break_id == LLDB_INVALID_BREAK_ID)
    s->PutCString("<invalid>");
  else if (m_location_id == LLDB_INVALID_BREAK_ID)
    s->Printf("%i", m_break_id);
  else
    s->Printf("%i.%i", m_break_id, m_location_id);
}

void BreakpointID::GetCanonicalReference(Stream *s, break_id_t bp_id,
                                         break_id_t loc_id) {
  if (bp_id == LLDB_INVALID_BREAK_ID)
    s->PutCString("<invalid>");
  else if (loc_id == LLDB_INVALID_BREAK_ID)
    s->Printf("%i", bp_id);
  else
    s->Printf("%i.%i", bp_id, loc_id);
}

bool BreakpointID::ParseCanonicalReference(const char *input,
                                           break_id_t *break_id_ptr,
                                           break_id_t *break_loc_id_ptr) {
  *break_id_ptr = LLDB_INVALID_BREAK_ID;
  *break_loc_id_ptr = LLDB_INVALID_BREAK_ID;

  if (input == nullptr || *input == '\0')
    return false;

  const char *format = "%i%n.%i%n";
  int chars_consumed_1 = 0;
  int chars_consumed_2 = 0;
  int n_items_parsed = ::sscanf(
      input, format,
      break_id_ptr,       // %i   parse the breakpoint ID
      &chars_consumed_1,  // %n   gets the number of characters parsed so far
      break_loc_id_ptr,   // %i   parse the breakpoint location ID
      &chars_consumed_2); // %n   gets the number of characters parsed so far

  if ((n_items_parsed == 1 && input[chars_consumed_1] == '\0') ||
      (n_items_parsed == 2 && input[chars_consumed_2] == '\0'))
    return true;

  // Badly formatted canonical reference.
  *break_id_ptr = LLDB_INVALID_BREAK_ID;
  *break_loc_id_ptr = LLDB_INVALID_BREAK_ID;
  return false;
}

bool BreakpointID::StringIsBreakpointName(const char *name, Error &error) {
  error.Clear();

  if (name && (name[0] >= 'A' && name[0] <= 'z')) {
    if (strcspn(name, ".- ") != strlen(name)) {
      error.SetErrorStringWithFormat("invalid breakpoint name: \"%s\"", name);
    }
    return true;
  } else
    return false;
}
