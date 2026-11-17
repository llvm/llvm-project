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
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr const char *DEFAULT_PROTOCOLS_PATH = "/etc/protocols";

#ifndef LIBC_COPT_PUBLIC_PACKAGING
static LIBC_THREAD_LOCAL const char *protocols_path = nullptr;

void netdb::set_protocols_path(const char *path) { protocols_path = path; }
#endif // LIBC_COPT_PUBLIC_PACKAGING

namespace {

enum class TokenKind : uint8_t {
  END_OF_LINE,
  TOKEN,
  EMBEDDED_NUL,
};

struct TokenResult {
  TokenKind kind;
  size_t len;
  bool matches;
  bool overflow;
};

struct AliasScan {
  size_t count = 0;
  size_t used = 0;
  bool matched = false;
  bool buf_overflow = false;
  bool valid = true;
};

enum class LineStatus : uint8_t {
  END_OF_FILE,
  SKIP,
  MATCHED,
};

struct ParsedMatch {
  LineStatus status;
  int proto_num = 0;
  size_t used = 0;
  size_t alias_count = 0;
  bool buf_overflow = false;
};

struct ChunkReader {
  static constexpr size_t CHUNK_SIZE = 256;

  File *file;
  char chunk[CHUNK_SIZE];
  size_t pos = 0;
  size_t len = 0;
  int error = 0;

  explicit ChunkReader(File *f) : file(f) {}

  bool next(char &ch) {
    if (pos == len) {
      FileIOResult res = file->read_unlocked(chunk, sizeof(chunk));
      if (res.has_error()) {
        error = res.error;
        return false;
      }
      if (res.value == 0)
        return false;

      pos = 0;
      len = res.value;
    }

    ch = chunk[pos++];
    return true;
  }
};

struct TokenWriter {
  cpp::string_view target;
  cpp::span<char> dst;
  size_t len = 0;
  bool matches = true;

  void push(char ch) {
    if (len < target.size()) {
      if (ch != target[len])
        matches = false;
    } else {
      matches = false;
    }

    if (len < dst.size())
      dst[len] = ch;
    ++len;
  }

  TokenResult finish() {
    const bool OVERFLOWED = len >= dst.size();
    if (!OVERFLOWED)
      dst[len] = '\0';

    bool is_match = matches;
    if (len != target.size())
      is_match = false;

    return {TokenKind::TOKEN, len, is_match, OVERFLOWED};
  }
};

} // namespace

static void drain_to_newline(ChunkReader &reader) {
  char ch = '\0';
  while (reader.next(ch)) {
    if (ch == '\n')
      return;
  }
}

static bool find_token_start(ChunkReader &reader, bool &saw_bytes, bool &at_eol,
                             char &ch) {
  while (reader.next(ch)) {
    saw_bytes = true;
    if (ch == '\n') {
      at_eol = true;
      return false;
    }
    if (ch == '#') {
      drain_to_newline(reader);
      at_eol = true;
      return false;
    }
    if (!internal::isspace(ch))
      return true;
  }

  at_eol = true;
  return false;
}

static bool is_token_end(ChunkReader &reader, char ch, bool &at_eol) {
  if (ch == '\n') {
    at_eol = true;
    return true;
  }
  if (ch == '#') {
    drain_to_newline(reader);
    at_eol = true;
    return true;
  }
  return internal::isspace(ch);
}

static TokenResult read_token(ChunkReader &reader, cpp::string_view target,
                              cpp::span<char> dst, bool &saw_bytes,
                              bool &at_eol) {
  char ch = '\0';
  if (!find_token_start(reader, saw_bytes, at_eol, ch))
    return {TokenKind::END_OF_LINE, 0, false, false};

  TokenWriter writer{target, dst};
  while (true) {
    if (ch == '\0') {
      drain_to_newline(reader);
      at_eol = true;
      return {TokenKind::EMBEDDED_NUL, 0, false, false};
    }
    writer.push(ch);

    if (!reader.next(ch)) {
      at_eol = true;
      break;
    }
    if (is_token_end(reader, ch, at_eol))
      break;
  }

  return writer.finish();
}

static constexpr cpp::optional<int>
parse_protocol_number(cpp::string_view token) {
  if (!internal::isdigit(token.front()))
    return cpp::nullopt;

  auto parsed = internal::strtointeger<int>(token.data(), 10, token.size());
  if (parsed.has_error() ||
      static_cast<size_t>(parsed.parsed_len) != token.size())
    return cpp::nullopt;

  return parsed.value;
}

static cpp::optional<int> read_protocol_number(ChunkReader &reader,
                                               bool &saw_bytes, bool &at_eol) {
  if (at_eol)
    return cpp::nullopt;

  constexpr size_t NUM_BUF_SIZE = 32;
  char num_buf[NUM_BUF_SIZE];
  const TokenResult NUM_TOK =
      read_token(reader, {}, num_buf, saw_bytes, at_eol);
  if (NUM_TOK.kind != TokenKind::TOKEN)
    return cpp::nullopt;

  cpp::optional<int> proto_num;
  if (!NUM_TOK.overflow)
    proto_num = parse_protocol_number({num_buf, NUM_TOK.len});
  if (!proto_num.has_value() && !at_eol)
    drain_to_newline(reader);

  return proto_num;
}

static AliasScan read_aliases(ChunkReader &reader, cpp::string_view target_name,
                              cpp::span<char> buffer, size_t initial_used,
                              bool initial_overflow, bool initial_matched,
                              bool &saw_bytes, bool &at_eol) {
  AliasScan scan{0, initial_used, initial_matched, initial_overflow, true};

  while (!at_eol) {
    cpp::span<char> dst;
    if (!scan.buf_overflow)
      dst = buffer.subspan(scan.used);

    const TokenResult ALIAS_TOK =
        read_token(reader, target_name, dst, saw_bytes, at_eol);
    if (ALIAS_TOK.kind == TokenKind::END_OF_LINE)
      break;
    if (ALIAS_TOK.kind == TokenKind::EMBEDDED_NUL) {
      scan.valid = false;
      return scan;
    }

    ++scan.count;
    if (ALIAS_TOK.matches)
      scan.matched = true;
    if (ALIAS_TOK.overflow)
      scan.buf_overflow = true;
    else
      scan.used += ALIAS_TOK.len + 1;
  }

  return scan;
}

static ParsedMatch parse_line(ChunkReader &reader, cpp::string_view target_name,
                              cpp::span<char> buffer) {
  bool saw_bytes = false;
  bool at_eol = false;
  const TokenResult PROTO_TOK =
      read_token(reader, target_name, buffer, saw_bytes, at_eol);
  if (PROTO_TOK.kind != TokenKind::TOKEN)
    return {saw_bytes ? LineStatus::SKIP : LineStatus::END_OF_FILE};

  const cpp::optional<int> PROTO_NUM =
      read_protocol_number(reader, saw_bytes, at_eol);
  if (!PROTO_NUM.has_value())
    return {LineStatus::SKIP};

  size_t initial_used = 0;
  if (!PROTO_TOK.overflow)
    initial_used = PROTO_TOK.len + 1;

  const AliasScan ALIASES =
      read_aliases(reader, target_name, buffer, initial_used,
                   PROTO_TOK.overflow, PROTO_TOK.matches, saw_bytes, at_eol);
  if (!ALIASES.valid || !ALIASES.matched)
    return {LineStatus::SKIP};

  return {LineStatus::MATCHED, *PROTO_NUM, ALIASES.used, ALIASES.count,
          ALIASES.buf_overflow};
}

static int pack_protoent(const ParsedMatch &match, cpp::span<char> buffer,
                         struct protoent *result_buf,
                         struct protoent **result) {
  if (match.buf_overflow)
    return ERANGE;

  cpp::span<char> tail = buffer.subspan(match.used);
  const uintptr_t TAIL_ADDR = reinterpret_cast<uintptr_t>(tail.data());
  const size_t PADDING =
      (alignof(char *) - TAIL_ADDR % alignof(char *)) % alignof(char *);
  if (tail.size() < PADDING)
    return ERANGE;

  tail = tail.subspan(PADDING);
  if (tail.size() / sizeof(char *) < match.alias_count + 1)
    return ERANGE;

  char **aliases = reinterpret_cast<char **>(tail.data());
  char *cursor = buffer.data();
  result_buf->p_name = cursor;

  for (size_t i = 0; i < match.alias_count; ++i) {
    while (*cursor != '\0')
      ++cursor;
    ++cursor;
    aliases[i] = cursor;
  }

  aliases[match.alias_count] = nullptr;
  result_buf->p_aliases = aliases;
  result_buf->p_proto = match.proto_num;
  *result = result_buf;
  return 0;
}

static int search_protocols(File *file, cpp::string_view target_name,
                            cpp::span<char> buffer, struct protoent *result_buf,
                            struct protoent **result) {
  ChunkReader reader(file);

  while (true) {
    const ParsedMatch LINE = parse_line(reader, target_name, buffer);
    if (reader.error != 0)
      return reader.error;
    if (LINE.status == LineStatus::END_OF_FILE)
      return 0;
    if (LINE.status == LineStatus::MATCHED)
      return pack_protoent(LINE, buffer, result_buf, result);
  }
}

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
  const cpp::string_view TARGET_NAME(name);

  const char *db_path = DEFAULT_PROTOCOLS_PATH;
#ifndef LIBC_COPT_PUBLIC_PACKAGING
  if (protocols_path != nullptr)
    db_path = protocols_path;
#endif

  auto file_or = openfile(db_path, "re");
  if (!file_or.has_value())
    return file_or.error() == ENOENT ? 0 : file_or.error();

  File *file = file_or.value();
  cpp::scope_exit close_file([file]() { file->close(); });
  File::FileLock lock(file);

  return search_protocols(file, TARGET_NAME, {buf, buflen}, result_buf, result);
}

} // namespace LIBC_NAMESPACE_DECL
