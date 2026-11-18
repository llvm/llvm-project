//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getprotobyname_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_protoent.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/netdb/getproto_internal.h"
#include "src/netdb/getprotobyname_r.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

namespace {

class ScopedProtocolsFile {
  char path[256];

public:
  ScopedProtocolsFile(const char *file_path,
                      LIBC_NAMESPACE::cpp::string_view content) {
    LIBC_NAMESPACE::internal::strlcpy(path, file_path, sizeof(path));

    LIBC_NAMESPACE::File *file = LIBC_NAMESPACE::openfile(path, "w").value();
    file->write(content.data(), content.size());
    file->close();

    LIBC_NAMESPACE::netdb::set_protocols_path(path);
  }

  ~ScopedProtocolsFile() {
    LIBC_NAMESPACE::netdb::set_protocols_path(nullptr);
    LIBC_NAMESPACE::remove(path);
  }

  const char *get_path() const { return path; }
};

constexpr void append_str(char *dst, size_t &pos,
                          LIBC_NAMESPACE::cpp::string_view text) {
  for (size_t i = 0; i < text.size(); ++i)
    dst[pos++] = text[i];
}

constexpr void make_indexed_alias(char (&out)[16], size_t index) {
  constexpr LIBC_NAMESPACE::cpp::string_view PREFIX = "long_alias_";
  for (size_t i = 0; i < PREFIX.size(); ++i)
    out[i] = PREFIX[i];
  out[PREFIX.size()] = static_cast<char>('0' + (index / 10));
  out[PREFIX.size() + 1] = static_cast<char>('0' + (index % 10));
  out[PREFIX.size() + 2] = '\0';
}

} // namespace

class LlvmLibcGetprotobynameRTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void TearDown() override {
    LIBC_NAMESPACE::netdb::set_protocols_path(nullptr);
    ErrnoCheckingTest::TearDown();
  }
};

TEST_F(LlvmLibcGetprotobynameRTest, CanonicalAndAliasLookups) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT =
      "ip\t0\tIP\t# internet protocol\n"
      "icmp\t1\tICMP\n"
      "tcp\t6\tTCP\n"
      "udp\t17\tUDP#adjacent comment\n"
      "gre\t47\n"
      "sctp\t132\tSCTP StreamControl sctp-alt\n"
      "primary\t200\tshadowed\n"
      "shadowed\t201\tSHADOWED\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_aliases.test"), CONTENT);

  struct protoent entry{};
  char buffer[256];
  struct protoent *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases, nullptr);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "TCP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases, nullptr);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "TCP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("UDP", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "udp");
  EXPECT_EQ(entry.p_proto, 17);
  ASSERT_NE(entry.p_aliases, nullptr);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "UDP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("sctp-alt", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "sctp");
  EXPECT_EQ(entry.p_proto, 132);
  ASSERT_NE(entry.p_aliases, nullptr);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  ASSERT_NE(entry.p_aliases[1], nullptr);
  ASSERT_NE(entry.p_aliases[2], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "SCTP");
  EXPECT_STREQ(entry.p_aliases[1], "StreamControl");
  EXPECT_STREQ(entry.p_aliases[2], "sctp-alt");
  EXPECT_EQ(entry.p_aliases[3], nullptr);

  // Earlier alias match takes precedence over a later canonical entry.
  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("shadowed", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "primary");
  EXPECT_EQ(entry.p_proto, 200);
  ASSERT_NE(entry.p_aliases, nullptr);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "shadowed");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  // Strict prefixes and empty names must not match.
  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tc", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp6", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("", &entry, buffer, sizeof(buffer),
                                             &result),
            0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, BufferReuseAcrossVaryingAliasCounts) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT =
      "sctp 132 SCTP sctp1 sctp2\n"
      "gre 47\n"
      "udp 17 UDP\n"
      "icmp 1 ICMP ping\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_reuse.test"), CONTENT);

  struct protoent entry{};
  char buffer[256];
  struct protoent *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("sctp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "sctp");
  ASSERT_NE(entry.p_aliases[2], nullptr);
  EXPECT_STREQ(entry.p_aliases[2], "sctp2");
  EXPECT_EQ(entry.p_aliases[3], nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("gre", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "gre");
  EXPECT_EQ(entry.p_proto, 47);
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("UDP", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "udp");
  EXPECT_EQ(entry.p_proto, 17);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "UDP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("missing", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("ping", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "icmp");
  EXPECT_EQ(entry.p_proto, 1);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  ASSERT_NE(entry.p_aliases[1], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "ICMP");
  EXPECT_STREQ(entry.p_aliases[1], "ping");
  EXPECT_EQ(entry.p_aliases[2], nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, OverlongRecordsAndManyAliases) {
  constexpr size_t LONG_NAME_LEN = 530;
  constexpr size_t WS_RUN_LEN = 200;
  constexpr size_t ALIAS_COUNT = 40;

  char content[2400];
  size_t pos = 0;

  // Record 1 (> 512B): overlong non-matching canonical name.
  for (size_t i = 0; i < LONG_NAME_LEN; ++i)
    content[pos++] = 'z';
  append_str(content, pos, " 99 fake_alias\n");

  // Record 2 (> 512B): 40 aliases (~570B), non-matching for "tcp"/"ip",
  // matching for "many" and "long_alias_39".
  append_str(content, pos, "many 250");
  for (size_t i = 0; i < ALIAS_COUNT; ++i) {
    char alias[16];
    make_indexed_alias(alias, i);
    content[pos++] = ' ';
    append_str(content, pos, LIBC_NAMESPACE::cpp::string_view(alias, 13));
  }
  content[pos++] = '\n';

  // Record 3 (> 512B): heavy horizontal whitespace around "tcp 6 TCP".
  for (size_t i = 0; i < WS_RUN_LEN; ++i)
    content[pos++] = ' ';
  append_str(content, pos, "tcp");
  for (size_t i = 0; i < WS_RUN_LEN; ++i)
    content[pos++] = '\t';
  append_str(content, pos, "6");
  for (size_t i = 0; i < WS_RUN_LEN; ++i)
    content[pos++] = ' ';
  append_str(content, pos, "TCP\r\n");

  // Blank, comment, and trailing-EOF records.
  append_str(content, pos, "\n   \t  \r\n# full comment line\nip 0");

  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_overlong.test"),
      LIBC_NAMESPACE::cpp::string_view(content, pos));

  struct protoent entry{};
  char small_buf[64];
  struct protoent *result = &entry;

  // >512B non-matching records drain cleanly with a 64B buffer.
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("nonexistent", &entry, small_buf,
                                             sizeof(small_buf), &result),
            0);
  EXPECT_EQ(result, nullptr);

  // >512B whitespace-padded "tcp ... 6 ... TCP" fits in a 64B buffer.
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, small_buf,
                                             sizeof(small_buf), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "TCP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  // Final entry at EOF without newline still parses cleanly.
  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, small_buf,
                                             sizeof(small_buf), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "ip");
  EXPECT_EQ(entry.p_proto, 0);
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  // >512B 40-alias match returns ERANGE with 64B buffer, succeeds with 1024B.
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("long_alias_39", &entry, small_buf,
                                             sizeof(small_buf), &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);

  char large_buf[1024];
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("long_alias_39", &entry, large_buf,
                                             sizeof(large_buf), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "many");
  EXPECT_EQ(entry.p_proto, 250);
  for (size_t i = 0; i < ALIAS_COUNT; ++i) {
    char expected_alias[16];
    make_indexed_alias(expected_alias, i);
    ASSERT_NE(entry.p_aliases[i], nullptr);
    EXPECT_STREQ(entry.p_aliases[i], expected_alias);
  }
  EXPECT_EQ(entry.p_aliases[ALIAS_COUNT], nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, EmbeddedNulHandling) {
  static constexpr char BAD_RECORDS[] = "\0tcp 6\n"
                                        "tc\0p 6\n"
                                        "tcp 6\0\n"
                                        "tcp 6 TCP bad\0alias\n";
  ScopedProtocolsFile bad_file(
      libc_make_test_file_path("getprotobyname_r_nul_bad.test"),
      LIBC_NAMESPACE::cpp::string_view(BAD_RECORDS, sizeof(BAD_RECORDS) - 1));

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = &entry;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  // Embedded NUL inside a '#' comment must be ignored.
  static constexpr char COMMENT_NUL[] = "\0tcp 99\n"
                                        "tcp 6 TCP # inline \0 comment\n"
                                        "# eof comment without newline";
  ScopedProtocolsFile ok_file(
      libc_make_test_file_path("getprotobyname_r_nul_comment.test"),
      LIBC_NAMESPACE::cpp::string_view(COMMENT_NUL, sizeof(COMMENT_NUL) - 1));

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "TCP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);

  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("missing", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, MalformedProtocolNumbers) {
  constexpr LIBC_NAMESPACE::cpp::string_view MALFORMED_CONTENT =
      "tcp\n"
      "tcp   \n"
      "tcp /\n"
      "tcp -1\n"
      "tcp 6x\n"
      "tcp 6x TCP\n"
      "tcp 9999999999\n"
      "tcp 00000000000000000000000000000006 TCP\n"
      "udp 17   ";
  ScopedProtocolsFile malformed_file(
      libc_make_test_file_path("getprotobyname_r_malformed.test"),
      MALFORMED_CONTENT);

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = &entry;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("udp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "udp");
  EXPECT_EQ(entry.p_proto, 17);

  constexpr LIBC_NAMESPACE::cpp::string_view FALLBACK_CONTENT =
      "tcp 00000000000000000000000000000006\n"
      "tcp 6x TCP\n"
      "tcp 6 TCP\n";
  ScopedProtocolsFile fallback_file(
      libc_make_test_file_path("getprotobyname_r_fallback.test"),
      FALLBACK_CONTENT);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
}

TEST_F(LlvmLibcGetprotobynameRTest, ResetPathWithNullptr) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT =
      "hermetic_test_proto_only 199\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_reset.test"), CONTENT);

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("hermetic_test_proto_only", &entry,
                                             buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_EQ(entry.p_proto, 199);

  // Passing nullptr restores the default "/etc/protocols" lookup path.
  LIBC_NAMESPACE::netdb::set_protocols_path(nullptr);
  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("hermetic_test_proto_only", &entry,
                                             buffer, sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, BufferBoundaryConditions) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT = "ip 0\n"
                                                       "tcp 6 TCP\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_erange.test"), CONTENT);

  struct protoent entry{};
  struct protoent *result = &entry;

  // (nullptr, 0) probe returns ERANGE on canonical/alias match and 0 on miss.
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, nullptr, 0, &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);

  result = &entry;
  EXPECT_EQ(
      LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, nullptr, 0, &result),
      ERANGE);
  EXPECT_EQ(result, nullptr);

  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("nonexistent", &entry, nullptr, 0,
                                             &result),
            0);
  EXPECT_EQ(result, nullptr);

  alignas(char *) char aligned_buf[64];

  // Name boundary: "ip\0" requires 3 bytes.
  result = &entry;
  EXPECT_EQ(
      LIBC_NAMESPACE::getprotobyname_r("ip", &entry, aligned_buf, 2, &result),
      ERANGE);
  EXPECT_EQ(result, nullptr);

  // Alignment padding boundary: "ip\0" fits in 3 bytes, leaving 0 for padding.
  result = &entry;
  EXPECT_EQ(
      LIBC_NAMESPACE::getprotobyname_r("ip", &entry, aligned_buf, 3, &result),
      ERANGE);
  EXPECT_EQ(result, nullptr);

  // Pointer-array boundary (0 aliases): 1 byte short of the null terminator.
  constexpr size_t IP_EXACT_LEN = alignof(char *) + sizeof(char *);
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, aligned_buf,
                                             IP_EXACT_LEN - 1, &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, aligned_buf,
                                             IP_EXACT_LEN, &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "ip");
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  // Alias text boundary: "tcp\0TCP\0" requires 8 bytes.
  result = &entry;
  EXPECT_EQ(
      LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, aligned_buf, 7, &result),
      ERANGE);
  EXPECT_EQ(result, nullptr);

  // Pointer-array boundary (1 alias): "tcp\0TCP\0" (8B) + 2 * sizeof(char *).
  constexpr size_t TCP_EXACT_LEN = 8 + 2 * sizeof(char *);
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, aligned_buf,
                                             TCP_EXACT_LEN - 1, &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("TCP", &entry, aligned_buf,
                                             TCP_EXACT_LEN, &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  ASSERT_NE(entry.p_aliases[0], nullptr);
  EXPECT_STREQ(entry.p_aliases[0], "TCP");
  EXPECT_EQ(entry.p_aliases[1], nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, IoErrors) {
  struct protoent entry{};
  char buffer[128];
  struct protoent *result = &entry;

  // Missing database file returns 0 with a null result pointer.
  const auto MISSING_PATH =
      libc_make_test_file_path("getprotobyname_r_missing.test");
  LIBC_NAMESPACE::netdb::set_protocols_path(MISSING_PATH);
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  // Path traversing through a regular file fails openfile with ENOTDIR.
  ScopedProtocolsFile regular_file(
      libc_make_test_file_path("getprotobyname_r_file.test"), "tcp 6\n");
  char notdir_path[280];
  const size_t BASE_LEN = LIBC_NAMESPACE::internal::strlcpy(
      notdir_path, regular_file.get_path(), sizeof(notdir_path));
  LIBC_NAMESPACE::internal::strlcpy(notdir_path + BASE_LEN, "/protocols",
                                    sizeof(notdir_path) - BASE_LEN);
  LIBC_NAMESPACE::netdb::set_protocols_path(notdir_path);

  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            ENOTDIR);
  EXPECT_EQ(result, nullptr);

  // Reading a directory fails read_unlocked with EISDIR.
  LIBC_NAMESPACE::netdb::set_protocols_path(".");
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            EISDIR);
  EXPECT_EQ(result, nullptr);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGetprotobynameRTest, NullptrCrash) {
  struct protoent entry{};
  char buffer[32];
  struct protoent *result = nullptr;

  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getprotobyname_r(nullptr, &entry, buffer,
                                         sizeof(buffer), &result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getprotobyname_r("tcp", nullptr, buffer, sizeof(buffer),
                                         &result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, nullptr, sizeof(buffer),
                                         &result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer, sizeof(buffer),
                                         nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
