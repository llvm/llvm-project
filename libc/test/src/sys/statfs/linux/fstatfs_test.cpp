#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/sys/statfs/fstatfs.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"
#include <linux/magic.h>
#include <llvm-libc-macros/linux/fcntl-macros.h>
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

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
  ASSERT_EQ(buf->f_type, static_cast<__kernel_long_t>(PROC_SUPER_MAGIC));
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/sys"), buf), Succeeds());
  ASSERT_EQ(buf->f_type, static_cast<__kernel_long_t>(SYSFS_MAGIC));
}

TEST(LlvmLibcSysStatfsTest, FstatfsNullBuffer) {
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(PathFD("/"), nullptr), Fails(EFAULT));
}

TEST(LlvmLibcSysStatfsTest, FstatfsInvalidFD) {
  ASSERT_THAT(LIBC_NAMESPACE::fstatfs(-1, nullptr), Fails(EBADF));
}
