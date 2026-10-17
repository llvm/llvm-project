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
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/func/realloc.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/setgroups.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
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

// Tracked alongside the database's own copy because the reentrant lookups open
// their own scoped database rather than sharing the iteration stream.
const char *group_file_path = LIBC_COPT_GROUP_FILE_PATH;

LIBC_CONSTINIT pwd::FlatFileDatabase<struct group>
    db(LIBC_COPT_GROUP_FILE_PATH);
// Note: These static buffers are process-global and NOT protected by a mutex
// at this stage. POSIX getgrent is non-reentrant.
//
// getgrent, getgrnam, and getgrgid share a single static buffer and struct
// group, as allowed by POSIX. The buffer grows as needed to fit the largest
// record read so far, and is reused without shrinking. endgrent() closes the
// file stream without freeing the buffer so that pointers returned before
// endgrent() remain valid until the next non-reentrant call.
LIBC_CONSTINIT pwd::DynamicBuffer line_buffer;
LIBC_CONSTINIT struct group grp_entry = {};

// The lookups are shared between the caller-supplied fixed buffer used by the
// reentrant entrypoints and the process-global growable buffer used by the
// non-reentrant ones. Both scan a scoped stream of their own so that a lookup
// does not disturb an in-progress getgrent iteration.
template <typename BufferType>
ErrorOr<bool> lookup_by_name(cpp::string_view name, struct group *grp,
                             BufferType &buffer, const char *path) {
  pwd::ScopedFlatFileDatabase<struct group> local_db(path);
  const auto matcher = [name](const struct group &entry) {
    return cpp::string_view(entry.gr_name) == name;
  };
  return local_db.lookup(matcher, grp, buffer);
}

template <typename BufferType>
ErrorOr<bool> lookup_by_gid(gid_t gid, struct group *grp, BufferType &buffer,
                            const char *path) {
  pwd::ScopedFlatFileDatabase<struct group> local_db(path);
  const auto matcher = [gid](const struct group &entry) {
    return entry.gr_gid == gid;
  };
  return local_db.lookup(matcher, grp, buffer);
}

// Small-buffer-optimised container for collecting group IDs without duplicates.
// Initial storage is stack-local; falls back to dynamic allocation if the
// user belongs to more than 32 groups.
class GidList {
  static constexpr size_t STATIC_CAP = 32;
  cpp::array<gid_t, STATIC_CAP> static_buf = {};
  gid_t *buf = static_buf.data();
  size_t count = 0;
  size_t cap = STATIC_CAP;

public:
  LIBC_INLINE GidList() = default;
  LIBC_INLINE ~GidList() {
    if (buf != static_buf.data())
      ::free(buf);
  }

  GidList(const GidList &) = delete;
  GidList &operator=(const GidList &) = delete;

  [[nodiscard]] LIBC_INLINE cpp::span<const gid_t> span() const {
    return {buf, count};
  }

  [[nodiscard]] LIBC_INLINE bool contains(gid_t gid) const {
    for (gid_t g : span()) {
      if (g == gid)
        return true;
    }
    return false;
  }

  [[nodiscard]] LIBC_INLINE bool push_back(gid_t gid) {
    if (count == cap) {
      if (LIBC_UNLIKELY(cap > cpp::numeric_limits<size_t>::max() /
                                  (2 * sizeof(gid_t))))
        return false;
      size_t new_cap = cap * 2;
      void *new_buf = nullptr;
      if (buf == static_buf.data()) {
        new_buf = ::malloc(new_cap * sizeof(gid_t));
        if (!new_buf)
          return false;
        for (size_t i = 0; i < count; ++i)
          static_cast<gid_t *>(new_buf)[i] = static_buf[i];
      } else {
        new_buf = ::realloc(buf, new_cap * sizeof(gid_t));
        if (!new_buf)
          return false;
      }
      buf = static_cast<gid_t *>(new_buf);
      cap = new_cap;
    }
    buf[count++] = gid;
    return true;
  }

  [[nodiscard]] LIBC_INLINE size_t size() const { return count; }

  [[nodiscard]] LIBC_INLINE const gid_t *data() const { return buf; }

  [[nodiscard]] LIBC_INLINE gid_t operator[](size_t i) const {
    LIBC_ASSERT(i < count);
    return buf[i];
  }
};

ErrorOr<void> populate_gid_list(cpp::string_view user, gid_t group,
                                GidList &gid_list, const char *path) {
  if (!gid_list.push_back(group))
    return Error(ENOMEM);

  pwd::ScopedFlatFileDatabase<struct group> local_db(path ? path
                                                          : group_file_path);
  pwd::ScopedDynamicBuffer buffer;
  struct group entry = {};

  const auto open_res = local_db.setdb();
  if (open_res.has_value()) {
    while (true) {
      const auto next_res = local_db.getnext(&entry, buffer);
      if (!next_res.has_value()) {
        if (next_res.error() == ENOMEM)
          return Error(ENOMEM);
        break;
      }
      if (!next_res.value())
        break;

      bool is_member = false;
      if (entry.gr_mem) {
        for (char **m = entry.gr_mem; *m != nullptr; ++m) {
          if (cpp::string_view(*m) == user) {
            is_member = true;
            break;
          }
        }
      }

      if (is_member && !gid_list.contains(entry.gr_gid)) {
        if (!gid_list.push_back(entry.gr_gid))
          return Error(ENOMEM);
      }
    }
  }

  return {};
}

} // namespace

void TESTONLY_set_group_path(const char *path) {
  close();
  line_buffer.release();
  group_file_path = path ? path : LIBC_COPT_GROUP_FILE_PATH;
  db.set_path(group_file_path);
}

void TESTONLY_reset_group_path() {
  close();
  line_buffer.release();
  group_file_path = LIBC_COPT_GROUP_FILE_PATH;
  db.set_path(group_file_path);
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

ErrorOr<bool> find_by_name(cpp::string_view name, struct group *grp,
                           cpp::span<char> buffer, const char *path) {
  return lookup_by_name(name, grp, buffer, path ? path : group_file_path);
}

ErrorOr<bool> find_by_gid(gid_t gid, struct group *grp, cpp::span<char> buffer,
                          const char *path) {
  return lookup_by_gid(gid, grp, buffer, path ? path : group_file_path);
}

ErrorOr<struct group *> find_by_name(cpp::string_view name) {
  const auto res =
      lookup_by_name(name, &grp_entry, line_buffer, group_file_path);
  if (!res.has_value())
    return Error(res.error());
  if (!res.value())
    return nullptr;
  return &grp_entry;
}

ErrorOr<struct group *> find_by_gid(gid_t gid) {
  const auto res = lookup_by_gid(gid, &grp_entry, line_buffer, group_file_path);
  if (!res.has_value())
    return Error(res.error());
  if (!res.value())
    return nullptr;
  return &grp_entry;
}

ErrorOr<size_t> get_group_list(cpp::string_view user, gid_t group,
                               gid_t *groups, size_t ngroups,
                               const char *path) {
  GidList gid_list;
  const auto res = populate_gid_list(user, group, gid_list, path);
  if (!res.has_value())
    return Error(res.error());

  const size_t copy_count =
      ngroups < gid_list.size() ? ngroups : gid_list.size();
  for (size_t i = 0; i < copy_count; ++i)
    groups[i] = gid_list[i];

  return gid_list.size();
}

ErrorOr<int> init_groups(cpp::string_view user, gid_t group, const char *path) {
  GidList gid_list;
  const auto res = populate_gid_list(user, group, gid_list, path);
  if (!res.has_value())
    return Error(res.error());

  return linux_syscalls::setgroups(gid_list.size(), gid_list.data());
}

} // namespace grp
} // namespace LIBC_NAMESPACE_DECL
