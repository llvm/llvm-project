//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of helper functions and parser for grp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_GRP_GRP_UTILS_H
#define LLVM_LIBC_SRC_GRP_GRP_UTILS_H

#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/pwd/flat_file_db.h"

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

template <>
ErrorOr<void> parse_line<struct group>(cpp::span<char> line,
                                       cpp::span<char> scratch,
                                       struct group *grp);

} // namespace pwd

namespace grp {

// Parse a colon-separated group line into a struct group.
bool parse_group_line(cpp::span<char> line, struct group *grp,
                      cpp::span<char *> mem_ptrs);

void TESTONLY_set_group_path(const char *path);
void TESTONLY_reset_group_path();

ErrorOr<void> open();
ErrorOr<void> close();
ErrorOr<struct group *> read_next();

ErrorOr<bool> find_by_name(cpp::string_view name, struct group *grp,
                           cpp::span<char> buffer, const char *path = nullptr);

ErrorOr<bool> find_by_gid(gid_t gid, struct group *grp, cpp::span<char> buffer,
                          const char *path = nullptr);

ErrorOr<struct group *> find_by_name(cpp::string_view name);
ErrorOr<struct group *> find_by_gid(gid_t gid);

// Fills up to ngroups into the groups array and returns the total number of
// groups found for user, or an Error if reading the database failed.
ErrorOr<size_t> get_group_list(cpp::string_view user, gid_t group,
                               gid_t *groups, size_t ngroups,
                               const char *path = nullptr);

} // namespace grp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_GRP_GRP_UTILS_H
