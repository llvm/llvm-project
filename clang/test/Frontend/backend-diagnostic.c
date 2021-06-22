// REQUIRES: x86-registered-target
// Play around with backend reporting:
// _REGULAR_: Regular behavior, no warning switch enabled.
// _PROMOTE_: Promote warning to error.
// _IGNORE_: Drop backend warning.
//
// RUN: not %clang_cc1 %s -fwarn-stack-size=0 -no-integrated-as -S -o - -triple=i386-apple-darwin 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=REGULAR --check-prefix=ASM
// RUN: not %clang_cc1 %s -fwarn-stack-size=0 -no-integrated-as -S -o - -triple=i386-apple-darwin -Werror=frame-larger-than 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=PROMOTE --check-prefix=ASM
// RUN: not %clang_cc1 %s -fwarn-stack-size=0 -no-integrated-as -S -o - -triple=i386-apple-darwin -Wno-frame-larger-than 2> %t.err
// RUN: FileCheck < %t.err %s --check-prefix=IGNORE --check-prefix=ASM
//
// RUN: %clang_cc1 %s -S -o - -triple=i386-apple-darwin -verify -no-integrated-as

extern void doIt(char *);

// REGULAR: warning: stack frame size ([[#]]) exceeds limit ([[#]]) in function 'stackSizeWarning'
// PROMOTE: error: stack frame size ([[#]]) exceeds limit ([[#]]) in function 'stackSizeWarning'
// IGNORE-NOT: stack frame size ([[#]]) exceeds limit ([[#]]) in function 'stackSizeWarning'
void stackSizeWarning() {
  char buffer[80];
  doIt(buffer);
}

// ASM: inline assembly requires more registers than available
void inlineAsmError(int x0, int x1, int x2, int x3, int x4,
                    int x5, int x6, int x7, int x8, int x9) {
  __asm__("hello world": : "r" (x0),"r" (x1),"r" (x2),"r" (x3), // expected-error + {{inline assembly requires more registers than available}}
          "r" (x4),"r" (x5),"r" (x6),"r" (x7),"r" (x8),"r" (x9));
}
