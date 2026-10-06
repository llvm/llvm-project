// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify
// expected-no-diagnostics

// Declarations that are not the POSIX umask keep a zero builtin id, so the
// fortify check is skipped even for out-of-range constants:
//   * a file-local umask with internal linkage;
//   * in C++, a umask without C language linkage (e.g. a user's own overload).

static int umask(int m) { return m; }

void call_static_umask(void) {
  (void)umask(0xFFFF);
}

#ifdef __cplusplus
namespace user {
int umask(int);
void call(void) { (void)umask(0xFFFF); }
} // namespace user
#endif
