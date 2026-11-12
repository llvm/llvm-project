/// Check that stack frame size warnings behave the same when safe stack is enabled

// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -fwarn-stack-size=48 -S -o - -triple=i386-apple-darwin 2>&1 | FileCheck --check-prefix=REGULAR %s
// RUN: %clang_cc1 %s -fwarn-stack-size=1060 -S -o - -triple=i386-apple-darwin 2>&1 | FileCheck --check-prefix=IGNORE %s

// RUN: %clang_cc1 %s -fsanitize=safe-stack -fwarn-stack-size=48 -S -o - -triple=i386-apple-darwin 2>&1 | FileCheck --check-prefix=SAFESTACK %s
// RUN: %clang_cc1 %s -fsanitize=safe-stack -fwarn-stack-size=1060 -S -o - -triple=i386-apple-darwin 2>&1 | FileCheck --check-prefix=IGNORE %s

extern void init(char *buf, int size);
extern int use_buf(char *buf, int size);

// REGULAR: warning: stack frame size ([[#]]) exceeds limit ([[#]]) in 'stackSizeWarning'
// SAFESTACK: warning: stack frame size ([[#]]) exceeds limit ([[#]]) in 'stackSizeWarning'
// IGNORE-NOT: stack frame size ([[#]]) exceeds limit ([[#]]) in 'stackSizeWarning'
void stackSizeWarning() {
  char buf[1024];
  init(buf, sizeof(buf));
  if (buf[0] < 42)
    use_buf(buf, sizeof(buf));
}
