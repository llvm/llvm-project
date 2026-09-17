//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of helper functions and parser for grp.
///
//===----------------------------------------------------------------------===//

#include "src/grp/grp_utils.h"
#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/span.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/pwd/dynamic_buffer.h"
#include "src/__support/pwd/field_tokenizer.h"
#include "src/__support/pwd/flat_file_db.h"
#include "src/__support/str_to_integer.h"

#ifndef LIBC_COPT_GROUP_FILE_PATH
#define LIBC_COPT_GROUP_FILE_PATH "/etc/group"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace {

// TODO: Replace with cpp::count when available in
// src/__support/CPP/algorithm.h.
size_t count_group_members(cpp::span<const char> line) {
  size_t max_members = 1;
  for (char c : line) {
    if (c == ',')
      ++max_members;
  }
  return max_members;
}

// Parse fixed fields (name, passwd, gid).
bool parse_group_fields(cpp::span<char> line, struct group *grp,
                        cpp::span<char> *members_out) {
  if (line.empty() || !grp || !members_out)
    return false;

  pwd::FieldTokenizer tokenizer(line, ':');

  const auto name = tokenizer.next_field();
  if (!name || name->empty() || name->front() == '\0')
    return false;
  grp->gr_name = name->data();

  const auto passwd = tokenizer.next_field();
  if (!passwd)
    return false;
  grp->gr_passwd = passwd->data();

  const auto gid_str = tokenizer.next_field();
  if (!gid_str || gid_str->empty() || !internal::isdigit(gid_str->front()))
    return false;
  auto gid_res = internal::strtointeger<gid_t>(gid_str->data(), 10);
  if (gid_res.has_error() || gid_res.parsed_len <= 0 ||
      static_cast<size_t>(gid_res.parsed_len) + 1 != gid_str->size() ||
      (*gid_str)[gid_res.parsed_len] != '\0')
    return false;
  grp->gr_gid = gid_res.value;

  const auto members_field = tokenizer.next_field();
  if (!members_field)
    return false;

  // Trailing delimiters or fields are invalid.
  if (tokenizer.next_field())
    return false;

  *members_out = *members_field;
  return true;
}

} // namespace

namespace pwd {

template <>
ErrorOr<void> parse_line<struct group>(cpp::span<char> line,
                                       cpp::span<char> scratch,
                                       struct group *grp) {
  if (!grp || line.empty() || line.back() != '\0')
    return Error(EINVAL);

  // Line excluding terminating null byte.
  const cpp::span<char> text = line.first(line.size() - 1);
  for (const char c : text) {
    if (c == '\0')
      return Error(EINVAL);
  }

  const size_t max_members = count_group_members(text);
  const uintptr_t tail = reinterpret_cast<uintptr_t>(scratch.data());
  // Calculate padding needed to align scratch to alignof(char *) so
  // that gr_mem pointer array elements can be safely stored.
  const size_t pad =
      (alignof(char *) - tail % alignof(char *)) % alignof(char *);
  if (scratch.size() < pad)
    return Error(ERANGE);
  const size_t remaining_bytes = scratch.size() - pad;
  if (remaining_bytes / sizeof(char *) < max_members + 1)
    return Error(ERANGE);

  const cpp::span<char *> mem_ptrs(
      reinterpret_cast<char **>(scratch.subspan(pad).data()),
      remaining_bytes / sizeof(char *));
  if (!grp::parse_group_line(line, grp, mem_ptrs))
    return Error(EINVAL);
  return {};
}

} // namespace pwd

namespace grp {

bool parse_group_line(cpp::span<char> line, struct group *grp,
                      cpp::span<char *> mem_ptrs) {
  if (mem_ptrs.empty())
    return false;

  cpp::span<char> members_field;
  if (!parse_group_fields(line, grp, &members_field))
    return false;

  size_t member_count = 0;
  if (!members_field.empty() && members_field.front() != '\0') {
    pwd::FieldTokenizer member_tokenizer(members_field, ',');
    while (const auto member = member_tokenizer.next_field()) {
      if (member->empty() || member->front() == '\0')
        continue;
      if (member_count + 1 >= mem_ptrs.size())
        return false;
      mem_ptrs[member_count++] = member->data();
    }
  }

  if (member_count >= mem_ptrs.size())
    return false;
  mem_ptrs[member_count] = nullptr;
  grp->gr_mem = mem_ptrs.data();

  return true;
}

namespace {

LIBC_CONSTINIT pwd::FlatFileDatabase<struct group>
    db(LIBC_COPT_GROUP_FILE_PATH);
// Note: These static buffers are process-global and NOT protected by a mutex
// at this stage. POSIX getgrent is non-reentrant.
//
// A single static buffer is reused across non-reentrant group calls via
// realloc, growing only to the high-water mark of the largest record seen.
// endgrent() closes the file stream without freeing the buffer so that
// pointers returned prior to endgrent() remain valid until the next
// non-reentrant call.
LIBC_CONSTINIT pwd::DynamicBuffer line_buffer;
struct group grp_entry;

} // namespace

void TESTONLY_set_group_path(const char *path) {
  close();
  line_buffer.release();
  db.set_path(path ? path : LIBC_COPT_GROUP_FILE_PATH);
}

void TESTONLY_reset_group_path() {
  close();
  line_buffer.release();
  db.set_path(LIBC_COPT_GROUP_FILE_PATH);
}

ErrorOr<void> open() { return db.setdb(); }

ErrorOr<void> close() { return db.enddb(); }

ErrorOr<struct group *> read_next() {
  const auto res = db.getnext(&grp_entry, line_buffer);
  if (!res.has_value())
    return Error(res.error());
  if (!res.value())
    return nullptr;
  return &grp_entry;
}

} // namespace grp
} // namespace LIBC_NAMESPACE_DECL
