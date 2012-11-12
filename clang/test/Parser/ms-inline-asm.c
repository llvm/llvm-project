// RUN: %clang_cc1 %s -verify -fms-extensions

#define M __asm int 0x2c
#define M2 int

void t1(void) { M } // expected-warning {{MS-style inline assembly is not supported}}
void t2(void) { __asm int 0x2c } // expected-warning {{MS-style inline assembly is not supported}}
void t3(void) { __asm M2 0x2c } // expected-warning {{MS-style inline assembly is not supported}}
void t4(void) { __asm mov eax, fs:[0x10] } // expected-warning {{MS-style inline assembly is not supported}}
void t5() {
  __asm { // expected-warning {{MS-style inline assembly is not supported}}
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {} // expected-warning {{MS-style inline assembly is not supported}}
}
int t6() {
  __asm int 3 ; } comments for single-line asm // expected-warning {{MS-style inline assembly is not supported}}
  __asm {} // expected-warning {{MS-style inline assembly is not supported}}

  __asm int 4 // expected-warning {{MS-style inline assembly is not supported}}
  return 10;
}
void t7() {
  __asm { // expected-warning {{MS-style inline assembly is not supported}}
    push ebx
    mov ebx, 0x07
    pop ebx
  }
}
void t8() {
  __asm nop __asm nop __asm nop // expected-warning {{MS-style inline assembly is not supported}}
}
void t9() {
  __asm nop __asm nop ; __asm nop // expected-warning {{MS-style inline assembly is not supported}}
}
int t_fail() { // expected-note {{to match this}}
  __asm // expected-warning {{MS-style inline assembly is not supported}}
  __asm { // expected-warning {{MS-style inline assembly is not supported}} expected-error 3 {{expected}} expected-note {{to match this}}
