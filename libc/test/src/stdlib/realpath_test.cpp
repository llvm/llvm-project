//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for realpath.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/func/free.h"
#include "hdr/limits_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/OSUtil/path.h"
#include "src/__support/fixedvector.h"
#include "src/__support/libc_assert.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/fcntl/openat.h"
#include "src/stdlib/realpath.h"
#include "src/string/strdup.h"
#include "src/sys/random/getrandom.h"
#include "src/sys/stat/mkdirat.h"
#include "src/unistd/close.h"
#include "src/unistd/getpid.h"
#include "src/unistd/unlinkat.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

namespace cpp = LIBC_NAMESPACE::cpp;
namespace path = LIBC_NAMESPACE::path;
using LIBC_NAMESPACE::FixedVector;
using LIBC_NAMESPACE::testing::tlog;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

// This test assumes the following values, so fail early if they mismatch.
static_assert(path::SEPARATOR == '/');
static_assert(path::CURRENT_DIR_COMPONENT == ".");
static_assert(path::PARENT_DIR_COMPONENT == "..");

// Size of a path separator.
constexpr size_t PATH_SEP_SIZE = 1;

// Creates the given directory if it does not exist. Returns zero on success.
[[nodiscard]] int ensure_directory_exists(int dirfd, const char *path,
                                          mode_t mode = 0755) {
  if (LIBC_NAMESPACE::mkdirat(dirfd, path, mode) == 0)
    return 0;
  if (libc_errno == EEXIST)
    libc_errno = 0;

  if (libc_errno != 0) {
    tlog << "Failed to create temp directory: " << path << "\n";
    return -1;
  }
  return 0;
}

// A test directory that removes itself on destruction.
class TestDir {
  // The test directory's absolute path.
  cpp::string path;

  // File descriptor of the test directory.
  int fd = -1;

  // Files created in this directory tree.
  // Use a C string instead of cpp::string because FixedVector
  // only supports trivially destructable types.
  FixedVector<char *, 64> files;

  // Subdirectories created in this directory tree.
  FixedVector<char *, 64> dirs;

public:
  TestDir() = default;

  // Initializes this TestDir container with the given path.
  void initialize(cpp::string directory_path, int dirfd) {
    LIBC_ASSERT(this->fd == -1);
    this->path = directory_path;
    this->fd = dirfd;
  }

  ~TestDir() {
    if (path.empty())
      return;

    for (size_t i = 0; i < files.size(); i++) {
      LIBC_NAMESPACE::unlinkat(fd, files[i], 0);
      ::free(files[i]);
    }

    // Remove directories in reverse order so they'll be empty.
    for (size_t i = dirs.size(); i > 0; i--) {
      LIBC_NAMESPACE::unlinkat(fd, dirs[i - 1], AT_REMOVEDIR);
      ::free(dirs[i - 1]);
    }

    LIBC_NAMESPACE::close(fd);
    LIBC_NAMESPACE::unlinkat(AT_FDCWD, path.c_str(), AT_REMOVEDIR);
  }

  TestDir(TestDir &other) = delete;
  TestDir &operator=(TestDir &other) = delete;
  TestDir(TestDir &&other) = delete;
  TestDir &operator=(TestDir &&other) = delete;

  // Returns the absolute path of `relative_path` in this test directory.
  cpp::string absolute_path(cpp::string_view relative_path) const {
    return path + "/" + relative_path;
  }

  // Returns this test directory path as a C string.
  const char *c_str() const { return path.c_str(); }

  // Returns this test directory path as a string view.
  const cpp::string_view view() const { return path; }

  // Creates a directory relative to TestDir. Returns zero on success.
  [[nodiscard]] int mkdir(const char *relative_path, mode_t mode = 0755) {
    char *path = LIBC_NAMESPACE::strdup(relative_path);
    if (path == nullptr)
      return -1;

    if (!dirs.push_back(path)) {
      tlog << "Not enough space in TestDir::dirs_\n";
      return -1;
    }

    return ensure_directory_exists(fd, relative_path, mode);
  }

  // Creates an empty file relative to TestDir. Returns zero on success.
  [[nodiscard]] int touch(const char *relative_path, mode_t mode = 0644) {
    char *path = LIBC_NAMESPACE::strdup(relative_path);
    if (path == nullptr)
      return -1;

    if (!files.push_back(path)) {
      tlog << "Not enough space in TestDir::files_\n";
      return -1;
    }
    int newfd =
        LIBC_NAMESPACE::openat(fd, relative_path, O_RDONLY | O_CREAT, mode);
    if (newfd < 0)
      return -1;
    return LIBC_NAMESPACE::close(newfd);
  }
};

cpp::string unique_id() {
  cpp::string id;
  id += cpp::to_string(LIBC_NAMESPACE::getpid());
  id += ".";

  constexpr cpp::string_view alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
  uint8_t rand_bytes[16] = {};
  (void)LIBC_NAMESPACE::getrandom(rand_bytes, sizeof(rand_bytes), 0);
  for (size_t i = 0; i < sizeof(rand_bytes); i++)
    id += alphabet[rand_bytes[i] % alphabet.size()];

  return id;
}

class LlvmLibcRealpathTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  char *realpath_buffered(const char *path) {
    return LIBC_NAMESPACE::realpath(path, buf_);
  }

  char *realpath_buffered(const cpp::string &path) {
    return realpath_buffered(path.c_str());
  }

  // Creates a test directory in dst. Returns true if successful.
  //
  // While we would prefer to return cpp::optional<TestDir> here,
  // LLVM-libc's optional expects types to be trivially destructible.
  [[nodiscard]] bool create_test_dir(const char *name, TestDir &dst) {
    // Use /tmp instead of the typical libc_make_test_file_path to ensure
    // the path is absolute and does not contain symlinks.
    cpp::string test_dir_path = "/tmp/LlvmLibcRealpathTest.";
    test_dir_path += name;
    test_dir_path += ".";

    // Include a unique ID in case multiple builds of this test run at once.
    test_dir_path += unique_id();

    if (ensure_directory_exists(AT_FDCWD, test_dir_path.c_str()))
      return false;
    int fd = LIBC_NAMESPACE::openat(AT_FDCWD, test_dir_path.c_str(),
                                    O_RDONLY | O_DIRECTORY);
    if (fd < 0)
      return false;

    dst.initialize(cpp::move(test_dir_path), fd);
    return true;
  }

private:
  char buf_[PATH_MAX];
};

TEST_F(LlvmLibcRealpathTest, ErrorsWithInvalidArgIfNullPath) {
  ASSERT_EQ(realpath_buffered(nullptr), nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcRealpathTest, ErrorsWithNoEntryIfEmptyPath) {
  ASSERT_EQ(realpath_buffered(""), nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcRealpathTest, OkIfPathArgIsExactlyMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX-1.
  cpp::string s(PATH_MAX - 1, '/');
  for (size_t i = 1; i < s.size(); i += 2)
    s[i] = '.';

  ASSERT_STREQ(realpath_buffered(s), "/");
}

// Creates a test directory that has an absolute path with exactly
// `desired_size` characters.
[[nodiscard]] bool create_absolute_path_with_size(TestDir &test_dir,
                                                  size_t desired_size,
                                                  cpp::string &out) {
  if (desired_size < test_dir.view().size() + PATH_SEP_SIZE) {
    tlog << "Test directory is already too long in "
            "create_absolute_path_with_size: "
         << test_dir.view().size() << "\n";
    return false;
  }
  size_t remaining_size = desired_size - test_dir.view().size() - PATH_SEP_SIZE;

  cpp::string relative_path;
  relative_path.reserve(remaining_size);

  while (remaining_size != 0) {
    if (!relative_path.empty()) {
      relative_path += '/';
      remaining_size -= PATH_SEP_SIZE;
    }

    size_t component_size = NAME_MAX;
    if (component_size > remaining_size)
      component_size = remaining_size;

    // If adding a component of this size would leave us in a state where
    // we only have enough space for a separator, shorten the component.
    if (remaining_size - component_size == PATH_SEP_SIZE)
      component_size -= 1;

    for (size_t i = 0; i < component_size; ++i)
      relative_path += 'a';
    remaining_size -= component_size;

    if (test_dir.mkdir(relative_path.c_str()))
      return false;
  }

  out = test_dir.absolute_path(relative_path);

  if (out.size() != desired_size) {
    tlog << "Failed to create path of size=" << desired_size << "\n";
    return false;
  }
  return true;
}

TEST_F(LlvmLibcRealpathTest, OkIfResolvedPathIsExactlyMaxSize) {
  TestDir test_dir;
  ASSERT_TRUE(create_test_dir("OkIfResolvedPathIsExactlyMaxSize", test_dir));

  cpp::string path;
  ASSERT_TRUE(create_absolute_path_with_size(test_dir, PATH_MAX - 1, path));

  ASSERT_STREQ(realpath_buffered(path), path.c_str());
}

TEST_F(LlvmLibcRealpathTest, ErrorsWithNameTooLongIfPathArgExceedsMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX.
  cpp::string s(PATH_MAX, '/');
  for (size_t i = 1; i < s.size(); i += 2)
    s[i] = '.';

  ASSERT_EQ(realpath_buffered(s), nullptr);
  ASSERT_ERRNO_EQ(ENAMETOOLONG);
}

TEST_F(LlvmLibcRealpathTest, RootResolvesToRoot) {
  ASSERT_STREQ(realpath_buffered("/"), "/");
}

TEST_F(LlvmLibcRealpathTest, RootDotDotTraversalStaysAtRoot) {
  ASSERT_STREQ(realpath_buffered("/.."), "/");
}

TEST_F(LlvmLibcRealpathTest, SimpleAbsolutePath) {
  TestDir test_dir;
  ASSERT_TRUE(create_test_dir("SimpleAbsolutePath", test_dir));

  ASSERT_THAT(test_dir.mkdir("a"), Succeeds());
  ASSERT_THAT(test_dir.mkdir("a/b"), Succeeds());

  ASSERT_STREQ(realpath_buffered(test_dir.absolute_path("a/b")),
               test_dir.absolute_path("a/b").c_str());
}

TEST_F(LlvmLibcRealpathTest, DotDotTraversesParent) {
  TestDir test_dir;
  ASSERT_TRUE(create_test_dir("DotDotTraversesParent", test_dir));

  ASSERT_THAT(test_dir.mkdir("a"), Succeeds());
  ASSERT_THAT(test_dir.mkdir("a/b"), Succeeds());

  ASSERT_STREQ(realpath_buffered(test_dir.absolute_path("a/b/..")),
               test_dir.absolute_path("a").c_str());
}

TEST_F(LlvmLibcRealpathTest, DotTraversalIsNop) {
  TestDir test_dir;
  ASSERT_TRUE(create_test_dir("DotTraversalIsNop", test_dir));

  ASSERT_THAT(test_dir.mkdir("a"), Succeeds());
  ASSERT_THAT(test_dir.mkdir("a/b"), Succeeds());

  ASSERT_STREQ(realpath_buffered(test_dir.absolute_path("a/b/./")),
               test_dir.absolute_path("a/b").c_str());
}

TEST_F(LlvmLibcRealpathTest, ConsecutiveSeparatorsIgnored) {
  TestDir test_dir;
  ASSERT_TRUE(create_test_dir("ConsecutiveSeparatorsIgnored", test_dir));

  ASSERT_THAT(test_dir.mkdir("a"), Succeeds());

  ASSERT_STREQ(realpath_buffered(test_dir.absolute_path("a//..///a//")),
               test_dir.absolute_path("a").c_str());
}

TEST_F(LlvmLibcRealpathTest, AllocatesResultWhenBufferIsNull) {
  char *result = LIBC_NAMESPACE::realpath("/", nullptr);
  ASSERT_STREQ(result, "/");
  ::free(result);
}
