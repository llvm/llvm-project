//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getprotobyname_r.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/getprotobyname_r.h"

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_protoent.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/str_to_integer.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace netdb {

static LIBC_THREAD_LOCAL const char *protocols_path = "/etc/protocols";

void set_protocols_path(const char *path) { protocols_path = path; }

static constexpr cpp::string_view next_token(cpp::string_view &line) {
  while (!line.empty() && internal::isspace(line.front()))
    line.remove_prefix(1);

  size_t len = 0;
  while (len < line.size() && !internal::isspace(line[len]))
    ++len;

  const cpp::string_view TOKEN = line.substr(0, len);
  line.remove_prefix(len);
  return TOKEN;
}

static constexpr cpp::optional<int>
parse_protocol_number(cpp::string_view token) {
  if (token.empty() || !internal::isdigit(token.front()))
    return cpp::nullopt;

  auto parsed = internal::strtointeger<int>(token.data(), 10, token.size());
  if (parsed.has_error() ||
      static_cast<size_t>(parsed.parsed_len) != token.size())
    return cpp::nullopt;

  return parsed.value;
}

static int pack_protoent(cpp::string_view proto_name, int proto_num,
                         struct protoent *result_buf, cpp::span<char> buffer,
                         struct protoent **result) {
  if (buffer.size() <= proto_name.size())
    return ERANGE;

  inline_memcpy(buffer.data(), proto_name.data(), proto_name.size());
  buffer[proto_name.size()] = '\0';
  result_buf->p_name = buffer.data();

  buffer = buffer.subspan(proto_name.size() + 1);
  const uintptr_t TAIL = reinterpret_cast<uintptr_t>(buffer.data());
  const size_t PADDING =
      (alignof(char *) - TAIL % alignof(char *)) % alignof(char *);
  if (buffer.size() < PADDING + sizeof(char *))
    return ERANGE;

  result_buf->p_aliases =
      reinterpret_cast<char **>(buffer.subspan(PADDING).data());
  result_buf->p_aliases[0] = nullptr;
  result_buf->p_proto = proto_num;
  *result = result_buf;
  return 0;
}

} // namespace netdb

LLVM_LIBC_FUNCTION(int, getprotobyname_r,
                   (const char *__restrict name,
                    struct protoent *__restrict result_buf,
                    char *__restrict buf, size_t buflen,
                    struct protoent **__restrict result)) {
  LIBC_CRASH_ON_NULLPTR(name);
  LIBC_CRASH_ON_NULLPTR(result_buf);
  if (buflen > 0)
    LIBC_CRASH_ON_NULLPTR(buf);
  LIBC_CRASH_ON_NULLPTR(result);

  *result = nullptr;

  auto file_or = openfile(netdb::protocols_path, "r");
  if (!file_or.has_value())
    return file_or.error() == ENOENT ? 0 : file_or.error();

  File *file = file_or.value();
  cpp::scope_exit close_file([file]() { file->close(); });
  File::FileLock lock(file);

  constexpr size_t MAX_LINE_LEN = 512;
  char line_buf[MAX_LINE_LEN];

  while (true) {
    size_t len = 0;
    bool in_comment = false;
    bool overflow = false;
    bool saw_bytes = false;

    while (true) {
      char ch = '\0';
      FileIOResult read_res = file->read_unlocked(&ch, 1);
      if (read_res.has_error())
        return read_res.error;
      if (read_res.value == 0)
        break;

      saw_bytes = true;
      if (ch == '\n')
        break;

      if (in_comment || overflow)
        continue;

      if (ch == '#')
        in_comment = true;
      else if (len < sizeof(line_buf))
        line_buf[len++] = ch;
      else
        overflow = true;
    }

    if (!saw_bytes)
      return 0;

    if (overflow)
      continue;

    cpp::string_view line(line_buf, len);
    const cpp::string_view PROTO_NAME = netdb::next_token(line);
    const cpp::string_view NUM_TOKEN = netdb::next_token(line);
    if (PROTO_NAME != name)
      continue;

    cpp::optional<int> proto_num = netdb::parse_protocol_number(NUM_TOKEN);
    if (proto_num.has_value())
      return netdb::pack_protoent(PROTO_NAME, *proto_num, result_buf,
                                  {buf, buflen}, result);
  }
}

} // namespace LIBC_NAMESPACE_DECL
