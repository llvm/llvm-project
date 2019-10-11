// RUN: %check_clang_tidy %s hicpp-no-assembler %t

__asm__(".symver foo, bar@v");
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not use inline assembler in safety-critical code [hicpp-no-assembler]

static int s asm("spam");
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use inline assembler in safety-critical code [hicpp-no-assembler]

void f() {
  __asm("mov al, 2");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use inline assembler in safety-critical code [hicpp-no-assembler]
}
