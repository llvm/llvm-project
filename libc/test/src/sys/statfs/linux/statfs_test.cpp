#include "src/sys/statfs/statfs.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"
#include <linux/magic.h>
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcSysStatfsTest, StatfsBasic) {
  statfs buf[1];
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/", buf), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/proc", buf), Succeeds());
  ASSERT_EQ(buf->f_type, static_cast<__kernel_long_t>(PROC_SUPER_MAGIC));
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/sys", buf), Succeeds());
  ASSERT_EQ(buf->f_type, static_cast<__kernel_long_t>(SYSFS_MAGIC));
}

TEST(LlvmLibcSysStatfsTest, InvalidPath) {
  statfs buf[1];
  ASSERT_THAT(LIBC_NAMESPACE::statfs("", buf), Fails(ENOENT));
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/nonexistent", buf), Fails(ENOENT));
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/dev/null/whatever", buf),
              Fails(ENOTDIR));
  ASSERT_THAT(LIBC_NAMESPACE::statfs(nullptr, buf), Fails(EFAULT));
}

TEST(LlvmLibcSysStatfsTest, StatfsNullBuffer) {
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/", nullptr), Fails(EFAULT));
}

TEST(LlvmLibcSysStatfsTest, NameTooLong) {
  statfs buf[1];
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/", buf), Succeeds());
  char *name = static_cast<char *>(__builtin_alloca(buf->f_namelen + 3));
  name[0] = '/';
  name[buf->f_namelen + 2] = '\0';
  for (unsigned i = 1; i < buf->f_namelen + 2; ++i) {
    name[i] = 'a';
  }
  ASSERT_THAT(LIBC_NAMESPACE::statfs(name, buf), Fails(ENAMETOOLONG));
}
