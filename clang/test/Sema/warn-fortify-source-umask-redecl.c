// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DNON_INT_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DSTRUCT_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DSTRUCT_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DPOINTER_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DPOINTER_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DNARROW_MODE_T -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DNARROW_MODE_T -verify
// expected-no-diagnostics

#if defined(NON_INT_MODE_T)
enum class mode_t { V = 0xFFFF };
extern "C" mode_t umask(mode_t);
void call_user_umask(void) { umask(mode_t::V); }
#elif defined(STRUCT_MODE_T)
typedef struct { int x; } mode_t;
#ifdef __cplusplus
extern "C"
#endif
mode_t umask(mode_t);
void call_user_umask(mode_t m) { umask(m); }
#elif defined(POINTER_MODE_T)
typedef void *mode_t;
#ifdef __cplusplus
extern "C"
#endif
mode_t umask(mode_t);
void call_user_umask(mode_t m) { umask(m); }
#elif defined(NARROW_MODE_T)
typedef unsigned char mode_t;
#ifdef __cplusplus
extern "C"
#endif
mode_t umask(mode_t);
void call_user_umask(void) { umask(0); }
#else
// A user declaration of umask whose signature does not match mode_t(mode_t) is
// neither flagged as an incompatible library redeclaration nor recognized as
// the libc umask builtin.

typedef unsigned int mode_t;
extern int umask(int);

void call_user_umask(void) {
  umask(0644);
  umask(0xFFFF);
}
#endif
