// REQUIRES: x86-registered-target

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -o - %s 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -o - %s -DWARN 2>&1 | FileCheck %s --check-prefix=WARN

#ifndef WARN
// ERROR: error: call to 'banned' declared with 'error' attribute: do not call me
__attribute__((error("do not call me"))) void banned(void);
void caller(void) {
  banned();
}
#else
// WARN: warning: call to 'warned' declared with 'warning' attribute: please do not call me
__attribute__((warning("please do not call me"))) void warned(void);
void caller(void) {
  warned();
}
#endif
