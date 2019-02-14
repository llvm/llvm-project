// RUN: %clangxx %s -o %t && %run %t
// UNSUPPORTED: ios

#include <assert.h>
#include <grp.h>
#include <memory>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

std::string any_group;
const int N = 123456;

void Check(const char *str) {
  if (!str)
    return;
  assert(strlen(str) != N);
}

void Check(const passwd *result) {
  Check(result->pw_name);
  Check(result->pw_passwd);
  assert(result->pw_uid != N);
  assert(result->pw_gid != N);
#if !defined(__ANDROID__)
  Check(result->pw_gecos);
#endif
  Check(result->pw_dir);

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
  assert(result->pw_change != N);
  Check(result->pw_class);
  assert(result->pw_expire != N);
#endif

#if defined(__FreeBSD__)
  assert(result->pw_fields != N);
#endif

  // SunOS also has pw_age and pw_comment which are documented as unused.
}

void Check(const group *result) {
  Check(result->gr_name);
  Check(result->gr_passwd);
  assert(result->gr_gid != N);
  for (char **mem = result->gr_mem; *mem; ++mem)
    Check(*mem);
  if (any_group.empty())
    any_group = result->gr_name;
}

template <class T, class Fn, class... Args>
void test(Fn f, Args... args) {
  T *result = f(args...);
  Check(result);
}

template <class T, class Fn, class... Args>
void test_r(Fn f, Args... args) {
  T gr;
  T *result;
  char buff[10000];
  assert(!f(args..., &gr, buff, sizeof(buff), &result));
  Check(&gr);
  Check(result);
}

int main(int argc, const char *argv[]) {
  test<passwd>(&getpwuid, 0);
  test<passwd>(&getpwnam, "root");
  test<group>(&getgrgid, 0);
  test<group>(&getgrnam, any_group.c_str());

#if !defined(__ANDROID__)
  setpwent();
  test<passwd>(&getpwent);
  setgrent();
  test<group>(&getgrent);

#if !defined(__APPLE__)
  setpwent();
  test_r<passwd>(&getpwent_r);
  setgrent();
  test_r<group>(&getgrent_r);
#endif

  test_r<passwd>(&getpwuid_r, 0);
  test_r<passwd>(&getpwnam_r, "root");

  test_r<group>(&getgrgid_r, 0);
  test_r<group>(&getgrnam_r, any_group.c_str());

#if defined(__linux__)
  auto pwd_file = [] {
    return std::unique_ptr<FILE, decltype(&fclose)>(fopen("/etc/passwd", "r"),
                                                    &fclose);
  };
  auto gr_file = [] {
    return std::unique_ptr<FILE, decltype(&fclose)>(fopen("/etc/group", "r"),
                                                    &fclose);
  };
  test<passwd>(&fgetpwent, pwd_file().get());
  test<group>(&fgetgrent, gr_file().get());
  test_r<passwd>(&fgetpwent_r, pwd_file().get());
  test_r<group>(&fgetgrent_r, gr_file().get());
#endif

#endif // __ANDROID__

  return 0;
}
