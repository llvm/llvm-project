// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks
// Disabling gnu inline assembly should have no effect on this testcase
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks -fno-gnu-inline-asm

#define M __asm int 0x2c
#define M2 int

void t1(void) { M }
void t2(void) { __asm int 0x2c }
void t3(void) { __asm M2 0x2c }
void t4(void) { __asm mov eax, fs:[0x10] }
void t5() {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
}
int t6() {
  __asm int 3 ; } comments for single-line asm
  __asm {}

  __asm int 4
  return 10;
}
void t7() {
  __asm {
    push ebx
    mov ebx, 0x07
    pop ebx
  }
}
void t8() {
  __asm nop __asm nop __asm nop
}
void t9() {
  __asm nop __asm nop ; __asm nop
}
void t10() {
  __asm {
    mov eax, 0
    __asm {
      mov eax, 1
      {
        mov eax, 2
      }
    }
  }
}
void t11() {
  do { __asm mov eax, 0 __asm { __asm mov edx, 1 } } while(0);
}
void t12() {
  __asm jmp label // expected-error {{use of undeclared label 'label'}}
}
void t13() {
  __asm m{o}v eax, ebx // expected-error {{expected identifier}} expected-error {{use of undeclared label '{o}v eax, ebx'}}
}

int t_fail() { // expected-note {{to match this}}
  __asm 
  __asm { // expected-error 3 {{expected}} expected-note {{to match this}}
