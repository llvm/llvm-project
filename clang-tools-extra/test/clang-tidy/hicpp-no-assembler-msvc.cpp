// REQUIRES: system-windows
// FIXME: Re-enable test on windows (PR36855)
// UNSUPPORTED: system-windows
// RUN: %check_clang_tidy %s hicpp-no-assembler %t

void f() {
  _asm {
    mov al, 2;
    // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: do not use inline assembler in safety-critical code [hicpp-no-assembler]
  }
}
