// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DNARROW_UMASK -verify
// expected-no-diagnostics

// Declarations that are not the POSIX umask are skipped by the fortify check:
//   * a file-local umask with internal linkage (zero builtin id);
//   * in C++, a umask without C language linkage (zero builtin id);
//   * a umask whose integer parameter is narrower than 16 bits (GH244171).

#ifdef NARROW_UMASK
typedef unsigned char mode_t;
mode_t umask(mode_t);

void call_narrow_umask(void) {
  (void)umask(0);
  (void)umask(0xFF);
}
#else
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
#endif
