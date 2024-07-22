#include "llvm-libc-macros/linux/fcntl-macros.h"
#include "src/__support/macros/config.h"
#include "src/fcntl/open.h"
#include "src/sys/statvfs/fstatvfs.h"
#include "src/sys/statvfs/linux/statfs_utils.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include <linux/magic.h>
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

#ifdef SYS_statfs64
using StatFs = statfs64;
#else
using StatFs = statfs;
#endif

namespace LIBC_NAMESPACE_DECL {
static int fstatfs(int fd, StatFs *buf) {
  using namespace statfs_utils;
  if (cpp::optional<StatFs> result = linux_fstatfs(fd)) {
    *buf = *result;
    return 0;
  }
  return -1;
}
} // namespace LIBC_NAMESPACE_DECL

struct PathFD {
  int fd;
  explicit PathFD(const char *path)
      : fd(LIBC_NAMESPACE::open(path, O_CLOEXEC | O_PATH)) {}
  ~PathFD() { LIBC_NAMESPACE::close(fd); }
  operator int() const { return fd; }
};

TEST(LlvmLibcSysStatvfsTest, FstatfsBasic) {
  StatFs buf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/"), &buf), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/proc"), &buf), Succeeds());
  ASSERT_EQ(buf.f_type, static_cast<decltype(buf.f_type)>(PROC_SUPER_MAGIC));
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/sys"), &buf), Succeeds());
  ASSERT_EQ(buf.f_type, static_cast<decltype(buf.f_type)>(SYSFS_MAGIC));
}

TEST(LlvmLibcSysStatvfsTest, FstatvfsInvalidFD) {
  struct statvfs buf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatvfs(-1, &buf), Fails(EBADF));
}
