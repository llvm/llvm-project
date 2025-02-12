// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s  -o /dev/null

#define _JBLEN ((9 * 2) + 3 + 16)
typedef int sigjmp_buf[_JBLEN + 1];
int sigsetjmp(sigjmp_buf env, int savemask);
void bar(void);
sigjmp_buf B;
void foo(void) {
  sigsetjmp(B, 1);
  bar();
}
