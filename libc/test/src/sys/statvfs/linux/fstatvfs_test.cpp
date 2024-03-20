#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/sys/statvfs/fstatvfs.h"
#include "src/sys/statvfs/linux/statfs_utils.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include <linux/magic.h>
#include <llvm-libc-macros/linux/fcntl-macros.h>
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

namespace LIBC_NAMESPACE {
static int fstatfs(int fd, struct statfs *buf) {
  using namespace statfs_utils;
  if (cpp::optional<LinuxStatFs> result = linux_fstatfs(fd)) {
    *buf = *result;
    return 0;
  }
  return -1;
}
} // namespace LIBC_NAMESPACE

struct PathFD {
  int fd;
  explicit PathFD(LIBC_NAMESPACE::cpp::string_view path)
      : fd(LIBC_NAMESPACE::open(path.data(), O_CLOEXEC | O_PATH)) {}
  ~PathFD() { LIBC_NAMESPACE::close(fd); }
  operator int() const { return fd; }
};

TEST(LlvmLibcSysStatfsTest, FstatfsBasic) {
  statfs buf[1];
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/"), buf), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/proc"), buf), Succeeds());
  ASSERT_EQ(buf->f_type, static_cast<decltype(buf->f_type)>(PROC_SUPER_MAGIC));
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/sys"), buf), Succeeds());
  ASSERT_EQ(buf->f_type, static_cast<decltype(buf->f_type)>(SYSFS_MAGIC));
}

TEST(LlvmLibcSysStatfsTest, FstatfsNullBuffer) {
  ASSERT_THAT(LIBC_NAMESPACE::fstatvfs(PathFD("/"), nullptr), Fails(EFAULT));
}

TEST(LlvmLibcSysStatfsTest, FstatfsInvalidFD) {
  statvfs buf[1];
  ASSERT_THAT(LIBC_NAMESPACE::fstatvfs(-1, buf), Fails(EBADF));
}
