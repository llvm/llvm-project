// RUN: %clang_cc1 -triple riscv32-unknown-elf -emit-llvm -DCHECK_IR < %s| FileCheck %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -emit-llvm -DCHECK_IR < %s| FileCheck %s
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv64-unknown-elf -verify -fsyntax-only

#if defined(CHECK_IR)
// CHECK-LABEL:  @foo_user() #0
// CHECK: ret void
__attribute__((interrupt("user"))) void foo_user(void) {}
// CHECK-LABEL:  @foo_supervisor() #1
// CHECK: ret void
__attribute__((interrupt("supervisor"))) void foo_supervisor(void) {}
// CHECK-LABEL:  @foo_machine() #2
// CHECK: ret void
__attribute__((interrupt("machine"))) void foo_machine(void) {}
// CHECK-LABEL:  @foo_default() #2
// CHECK: ret void
__attribute__((interrupt())) void foo_default(void) {}
// CHECK-LABEL:  @foo_default2() #2
// CHECK: ret void
__attribute__((interrupt())) void foo_default2(void) {}
// CHECK: attributes #0
// CHECK: "interrupt"="user"
// CHECK: attributes #1
// CHECK: "interrupt"="supervisor"
// CHECK: attributes #2
// CHECK: "interrupt"="machine"
#else
struct a { int b; };

struct a test __attribute__((interrupt)); // expected-warning {{'interrupt' attribute only applies to functions}}

__attribute__((interrupt("USER"))) void foo1(void) {} // expected-warning {{'interrupt' attribute argument not supported: USER}}

__attribute__((interrupt("user", 1))) void foo2(void) {} // expected-error {{'interrupt' attribute takes no more than 1 argument}}

__attribute__((interrupt)) int foo3(void) {return 0;} // expected-warning {{RISC-V 'interrupt' attribute only applies to functions that have a 'void' return type}}

__attribute__((interrupt())) void foo4();
__attribute__((interrupt())) void foo4() {};

__attribute__((interrupt())) void foo5(int a) {} // expected-warning {{RISC-V 'interrupt' attribute only applies to functions that have no parameters}}

__attribute__((interrupt("user"), interrupt("supervisor"))) void foo6(void) {} // expected-warning {{repeated RISC-V 'interrupt' attribute}} \
  // expected-note {{repeated RISC-V 'interrupt' attribute is here}}

__attribute__((interrupt, interrupt)) void foo7(void) {} // expected-warning {{repeated RISC-V 'interrupt' attribute}} \
                                                     // expected-note {{repeated RISC-V 'interrupt' attribute is here}}

__attribute__((interrupt(""))) void foo8(void) {} // expected-warning {{'interrupt' attribute argument not supported}}

__attribute__((interrupt("user"))) void foo9(void);
__attribute__((interrupt("supervisor"))) void foo9(void);
__attribute__((interrupt("machine"))) void foo9(void);

__attribute__((interrupt("user"))) void foo10(void) {}
__attribute__((interrupt("supervisor"))) void foo11(void) {}
__attribute__((interrupt("machine"))) void foo12(void) {}
__attribute__((interrupt())) void foo13(void) {}
__attribute__((interrupt)) void foo14(void) {}
#endif

