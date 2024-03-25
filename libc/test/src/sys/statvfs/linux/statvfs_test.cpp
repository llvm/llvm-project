#include "src/sys/statvfs/linux/statfs_utils.h"
#include "src/sys/statvfs/statvfs.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include <linux/magic.h>
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

namespace LIBC_NAMESPACE {
static int statfs(const char *path, struct statfs *buf) {
  using namespace statfs_utils;
  if (cpp::optional<LinuxStatFs> result = linux_statfs(path)) {
    *buf = *result;
    return 0;
  }
  return -1;
}
} // namespace LIBC_NAMESPACE

TEST(LlvmLibcSysStatfsTest, StatfsBasic) {
  struct statfs buf;
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/", &buf), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/proc", &buf), Succeeds());
  ASSERT_EQ(buf.f_type, static_cast<decltype(buf.f_type)>(PROC_SUPER_MAGIC));
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/sys", &buf), Succeeds());
  ASSERT_EQ(buf.f_type, static_cast<decltype(buf.f_type)>(SYSFS_MAGIC));
}

TEST(LlvmLibcSysStatfsTest, StatvfsInvalidPath) {
  struct statvfs buf;
  ASSERT_THAT(LIBC_NAMESPACE::statvfs("", &buf), Fails(ENOENT));
  ASSERT_THAT(LIBC_NAMESPACE::statvfs("/nonexistent", &buf), Fails(ENOENT));
  ASSERT_THAT(LIBC_NAMESPACE::statvfs("/dev/null/whatever", &buf),
              Fails(ENOTDIR));
  ASSERT_THAT(LIBC_NAMESPACE::statvfs(nullptr, &buf), Fails(EFAULT));
}

TEST(LlvmLibcSysStatfsTest, StatvfsNameTooLong) {
  struct statvfs buf;
  ASSERT_THAT(LIBC_NAMESPACE::statvfs("/", &buf), Succeeds());
  char *name = static_cast<char *>(__builtin_alloca(buf.f_namemax + 3));
  name[0] = '/';
  name[buf.f_namemax + 2] = '\0';
  for (unsigned i = 1; i < buf.f_namemax + 2; ++i) {
    name[i] = 'a';
  }
  ASSERT_THAT(LIBC_NAMESPACE::statvfs(name, &buf), Fails(ENAMETOOLONG));
}
