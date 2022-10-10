// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks
// Disabling gnu inline assembly should have no effect on this testcase
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks -fno-gnu-inline-asm

#define M __asm int 0x2c
#define M2 int

void t1(void) { M }
void t2(void) { __asm int 2ch }
void t3(void) { __asm M2 2ch }
void t4(void) { __asm mov eax, fs:[10h] }
void t5(void) {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
}
int t6(void) {
  __asm int 3 ; } comments for single-line asm
  __asm {}

  __asm int 4
  return 10;
}
void t7(void) {
  __asm {
    push ebx
    mov ebx, 07h
    pop ebx
  }
}
void t8(void) {
  __asm nop __asm nop __asm nop
}
void t9(void) {
  __asm nop __asm nop ; __asm nop
}
void t10(void) {
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
void t11(void) {
  do { __asm mov eax, 0 __asm { __asm mov edx, 1 } } while(0);
}
void t12(void) {
  __asm jmp label // expected-error {{use of undeclared label 'label'}}
}
void t13(void) {
  __asm m{o}v eax, ebx // expected-error {{unknown token in expression}}
}

void t14(void) {
  enum { A = 1, B };
  __asm mov eax, offset A // expected-error {{offset operator cannot yet handle constants}}
}

// GH57791
typedef struct S {
  unsigned bf1:1; // expected-note {{bit-field is declared here}}
  unsigned bf2:1; // expected-note {{bit-field is declared here}}
} S;
void t15(S s) {
  __asm {
    mov eax, s.bf1 // expected-error {{an inline asm block cannot have an operand which is a bit-field}}
    mov s.bf2, eax // expected-error {{an inline asm block cannot have an operand which is a bit-field}}
  }
}

int t_fail(void) { // expected-note {{to match this}}
  __asm 
  __asm { // expected-error 3 {{expected}} expected-note {{to match this}}
