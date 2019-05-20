// RUN: %check_clang_tidy %s cert-err52-cpp %t

typedef void *jmp_buf;
extern int __setjmpimpl(jmp_buf);
#define setjmp(x) __setjmpimpl(x)
[[noreturn]] extern void longjmp(jmp_buf, int);

namespace std {
using ::jmp_buf;
using ::longjmp;
}

static jmp_buf env;
void g() {
  std::longjmp(env, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not call 'longjmp'; consider using exception handling instead [cert-err52-cpp]
  ::longjmp(env, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not call 'longjmp'; consider using exception handling instead
  longjmp(env, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not call 'longjmp'; consider using exception handling instead
}

void f() {
  (void)setjmp(env);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not call 'setjmp'; consider using exception handling instead
}
