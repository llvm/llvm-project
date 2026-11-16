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
#include "src/netdb/getprotobyname_r.h"
#include "src/stdio/remove.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

namespace {

constexpr const char *DEFAULT_PROTOCOLS_PATH = "/etc/protocols";

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
    LIBC_NAMESPACE::netdb::set_protocols_path(DEFAULT_PROTOCOLS_PATH);
    LIBC_NAMESPACE::remove(path);
  }

  const char *get_path() const { return path; }
};

} // namespace

class LlvmLibcGetprotobynameRTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  void TearDown() override {
    LIBC_NAMESPACE::netdb::set_protocols_path(DEFAULT_PROTOCOLS_PATH);
    ErrnoCheckingTest::TearDown();
  }
};

TEST_F(LlvmLibcGetprotobynameRTest, LookupSuccess) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT =
      "ip\t0\tIP\t# internet protocol\n"
      "icmp\t1\tICMP\n"
      "tcp\t6\tTCP\n"
      "udp\t17\tUDP\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_success.test"), CONTENT);

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "ip");
  EXPECT_EQ(entry.p_proto, 0);
  ASSERT_NE(entry.p_aliases, nullptr);
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases, nullptr);
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("udp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "udp");
  EXPECT_EQ(entry.p_proto, 17);
  ASSERT_NE(entry.p_aliases, nullptr);
  EXPECT_EQ(entry.p_aliases[0], nullptr);
}

TEST_F(LlvmLibcGetprotobynameRTest, CommentsBlankAndOverlongLines) {
  constexpr size_t PADDING_LEN = 520;
  constexpr LIBC_NAMESPACE::cpp::string_view TRAILER =
      " fake 7\n"
      "\n"
      "   \t  \r\n"
      "# full comment line\n"
      "tcp\t6 # inline comment\r\n"
      "ip 0";

  char content[PADDING_LEN + TRAILER.size()];
  for (size_t i = 0; i < PADDING_LEN; ++i)
    content[i] = 'z';
  for (size_t i = 0; i < TRAILER.size(); ++i)
    content[PADDING_LEN + i] = TRAILER[i];

  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_comments.test"),
      LIBC_NAMESPACE::cpp::string_view(content, sizeof(content)));

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = &entry;

  // Overlong lines (> 512 bytes) must be drained without injecting their tail.
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("fake", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
  ASSERT_NE(entry.p_aliases, nullptr);
  EXPECT_EQ(entry.p_aliases[0], nullptr);

  // Final entry without a trailing newline must still parse cleanly.
  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("ip", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "ip");
  EXPECT_EQ(entry.p_proto, 0);
}

TEST_F(LlvmLibcGetprotobynameRTest, NotFoundAndMalformedNumbers) {
  constexpr LIBC_NAMESPACE::cpp::string_view MALFORMED_CONTENT =
      "tcp\n"
      "tcp /\n"
      "tcp -1\n"
      "tcp 6x\n"
      "tcp 9999999999\n"
      "udp 17\n";
  ScopedProtocolsFile malformed_file(
      libc_make_test_file_path("getprotobyname_r_malformed.test"),
      MALFORMED_CONTENT);

  struct protoent entry{};
  char buffer[128];
  struct protoent *result = &entry;

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("nonexistent", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  result = &entry;
  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  EXPECT_EQ(result, nullptr);

  constexpr LIBC_NAMESPACE::cpp::string_view FALLBACK_CONTENT =
      "tcp\n"
      "tcp /\n"
      "tcp 6x\n"
      "tcp 9999999999\n"
      "tcp 6\n";
  ScopedProtocolsFile fallback_file(
      libc_make_test_file_path("getprotobyname_r_fallback.test"),
      FALLBACK_CONTENT);

  ASSERT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, buffer,
                                             sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &entry);
  EXPECT_STREQ(entry.p_name, "tcp");
  EXPECT_EQ(entry.p_proto, 6);
}

TEST_F(LlvmLibcGetprotobynameRTest, BufferTooSmall) {
  constexpr LIBC_NAMESPACE::cpp::string_view CONTENT = "tcp 6\n";
  ScopedProtocolsFile test_file(
      libc_make_test_file_path("getprotobyname_r_erange.test"), CONTENT);

  struct protoent entry{};
  struct protoent *result = &entry;

  // Zero-length buffer with nullptr must not crash and must return ERANGE.
  EXPECT_EQ(
      LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, nullptr, 0, &result),
      ERANGE);
  EXPECT_EQ(result, nullptr);

  // Buffer too small to hold the null-terminated protocol name ("tcp\0").
  char tiny_buf[3];
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, tiny_buf,
                                             sizeof(tiny_buf), &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);

  // Buffer holds "tcp\0" but lacks room for the aligned alias terminator.
  char short_buf[6];
  result = &entry;
  EXPECT_EQ(LIBC_NAMESPACE::getprotobyname_r("tcp", &entry, short_buf,
                                             sizeof(short_buf), &result),
            ERANGE);
  EXPECT_EQ(result, nullptr);
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
