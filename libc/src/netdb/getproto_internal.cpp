//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of shared /etc/protocols lookup helpers.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/getproto_internal.h"

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_protoent.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/File/file.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/memory_size.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr const char *DEFAULT_PROTOCOLS_PATH = "/etc/protocols";

namespace {

struct TokenMatcher {
  cpp::string_view target;
  size_t len = 0;
  bool matches = true;

  void push(char ch) {
    if (len >= target.size() || ch != target[len])
      matches = false;
    ++len;
  }

  bool matched() const { return matches && len == target.size(); }
};

class LineReader {
  File *file;
  bool saw_bytes = false;
  bool at_eol = false;
  bool at_eof = false;
  bool in_comment = false;
  bool invalid_line = false;

  ErrorOr<cpp::optional<char>> next_char() {
    while (true) {
      char ch = '\0';
      FileIOResult res = file->read_unlocked(&ch, 1);
      if (res.has_error())
        return Error(res.error);
      if (res.value == 0) {
        at_eof = true;
        at_eol = true;
        return cpp::optional<char>(cpp::nullopt);
      }

      saw_bytes = true;
      if (ch == '\n') {
        at_eol = true;
        return cpp::optional<char>(cpp::nullopt);
      }
      if (ch == '#')
        in_comment = true;
      if (in_comment)
        continue;
      if (ch == '\0')
        invalid_line = true;
      return cpp::optional<char>(ch);
    }
  }

public:
  explicit LineReader(File *f) : file(f) {}

  void begin_line() {
    saw_bytes = false;
    at_eol = at_eof;
    in_comment = false;
    invalid_line = false;
  }

  bool has_bytes() const { return saw_bytes; }

  bool is_invalid() const { return invalid_line; }

  ErrorOr<cpp::optional<bool>> read_token(cpp::StringStream &dst,
                                          cpp::string_view target) {
    TokenMatcher matcher{target};
    bool in_token = false;

    while (!at_eol) {
      ErrorOr<cpp::optional<char>> ch_or = next_char();
      if (!ch_or.has_value())
        return Error(ch_or.error());
      if (!ch_or->has_value())
        break;

      const char CH = **ch_or;
      if (internal::isspace(CH)) {
        if (in_token)
          break;
        continue;
      }

      in_token = true;
      matcher.push(CH);
      dst << CH;
    }

    if (!in_token)
      return cpp::optional<bool>(cpp::nullopt);

    dst << cpp::StringStream::ENDS;
    return cpp::optional<bool>(matcher.matched());
  }
};

struct LineScan {
  size_t token_count = 0;
  bool matched = false;
};

} // namespace

static cpp::optional<int>
parse_protocol_number(const cpp::StringStream &num_out) {
  if (num_out.overflow())
    return cpp::nullopt;

  cpp::string_view raw = num_out.str();
  if (raw.empty())
    return cpp::nullopt;

  cpp::string_view token = raw.substr(0, raw.size() - 1);
  if (!internal::isdigit(token.front()))
    return cpp::nullopt;

  auto parsed = internal::strtointeger<int>(token.data(), 10, token.size());
  if (parsed.has_error() ||
      static_cast<size_t>(parsed.parsed_len) != token.size())
    return cpp::nullopt;

  return parsed.value;
}

static ErrorOr<LineScan> scan_line_tokens(LineReader &reader,
                                          cpp::string_view target_name,
                                          cpp::StringStream &out,
                                          cpp::StringStream &num_out) {
  reader.begin_line();
  LineScan scan{};

  while (true) {
    cpp::StringStream *dst = &out;
    cpp::string_view tok_target = target_name;
    if (scan.token_count == 1) {
      dst = &num_out;
      tok_target = {};
    }

    ErrorOr<cpp::optional<bool>> tok_or = reader.read_token(*dst, tok_target);
    if (!tok_or.has_value())
      return Error(tok_or.error());
    if (!tok_or->has_value())
      return scan;

    if (**tok_or)
      scan.matched = true;
    ++scan.token_count;
  }
}

static int pack_protoent(const cpp::StringStream &out, cpp::span<char> buffer,
                         int proto_num, size_t alias_count,
                         struct protoent *result_buf,
                         struct protoent **result) {
  if (out.overflow())
    return ERANGE;

  cpp::span<char> tail = buffer.subspan(out.str().size());
  const uintptr_t TAIL_ADDR = reinterpret_cast<uintptr_t>(tail.data());
  const size_t PADDING =
      internal::SafeMemSize::offset_to(TAIL_ADDR, alignof(char *));
  if (tail.size() < PADDING)
    return ERANGE;

  tail = tail.subspan(PADDING);
  if (tail.size() / sizeof(char *) < alias_count + 1)
    return ERANGE;

  char **aliases = reinterpret_cast<char **>(tail.data());
  char *cursor = buffer.data();
  result_buf->p_name = cursor;

  for (size_t i = 0; i < alias_count; ++i) {
    while (*cursor != '\0')
      ++cursor;
    ++cursor;
    aliases[i] = cursor;
  }

  aliases[alias_count] = nullptr;
  result_buf->p_aliases = aliases;
  result_buf->p_proto = proto_num;
  *result = result_buf;
  return 0;
}

static ErrorOr<cpp::optional<int>> parse_line(LineReader &reader,
                                              cpp::string_view target_name,
                                              cpp::span<char> buffer,
                                              struct protoent *result_buf,
                                              struct protoent **result) {
  constexpr size_t NUM_BUF_SIZE = 32;
  char num_buf[NUM_BUF_SIZE];
  cpp::StringStream out(buffer);
  cpp::StringStream num_out(num_buf);

  ErrorOr<LineScan> scan_or =
      scan_line_tokens(reader, target_name, out, num_out);
  if (!scan_or.has_value())
    return Error(scan_or.error());

  if (scan_or->token_count == 0) {
    if (!reader.has_bytes())
      return cpp::optional<int>(0);
    return cpp::optional<int>(cpp::nullopt);
  }
  if (!scan_or->matched || reader.is_invalid())
    return cpp::optional<int>(cpp::nullopt);

  const cpp::optional<int> PROTO_NUM = parse_protocol_number(num_out);
  if (!PROTO_NUM.has_value())
    return cpp::optional<int>(cpp::nullopt);

  return cpp::optional<int>(pack_protoent(
      out, buffer, *PROTO_NUM, scan_or->token_count - 2, result_buf, result));
}

static int search_protocols(File *file, cpp::string_view target_name,
                            cpp::span<char> buffer, struct protoent *result_buf,
                            struct protoent **result) {
  LineReader reader(file);

  while (true) {
    ErrorOr<cpp::optional<int>> line_or =
        parse_line(reader, target_name, buffer, result_buf, result);
    if (!line_or.has_value())
      return line_or.error();
    if (line_or->has_value())
      return **line_or;
  }
}

int netdb::lookup_proto_by_name(cpp::string_view name,
                                struct protoent *result_buf,
                                cpp::span<char> buffer,
                                struct protoent **result, const char *path) {
  *result = nullptr;

  const char *db_path = path != nullptr ? path : DEFAULT_PROTOCOLS_PATH;
  auto file_or = openfile(db_path, "re");
  if (!file_or.has_value())
    return file_or.error() == ENOENT ? 0 : file_or.error();

  File *file = file_or.value();
  cpp::scope_exit close_file([file]() { file->close(); });
  File::FileLock lock(file);

  return search_protocols(file, name, buffer, result_buf, result);
}

} // namespace LIBC_NAMESPACE_DECL
