#include <stdio.h>

typedef int (*func_ptr)(int, int);

int add(int a, int b) { return a + b; }

int getConst(int a, int b) { return 0xaa55; }

void foo() {
  // clang-format off
  __asm__ __volatile("stp x29, x30, [sp, #-16]!\n"
                     "adrp x0, getConst\n"
                     "add x0, x0, :lo12:getConst\n"
                     "blr x0\n"
                     "ldp x29, x30, [sp], #16\n"
                     :::);
  // clang-format on
  return;
}

int main() {
  func_ptr fun;
  fun = add;
  int sum = fun(10, 20); // indirect call to 'add'
  printf("The sum is: %d\n", sum);
  foo();
  return 0;
}
/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q -no-pie -fpie
RUN: llvm-objdump --disassemble-symbols=main %t.exe \
RUN:   | FileCheck %s --check-prefix=CHECKINDIRECTREG

CHECKINDIRECTREG: mov w0, #0xa
CHECKINDIRECTREG-NEXT: mov w1, #0x14
CHECKINDIRECTREG-NEXT: blr x8

RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
RUN:   -o %t.instrumented \
RUN:   | FileCheck %s --check-prefix=CHECK-LOG

CHECK-LOG-NOT: BOLT-INSTRUMENTER: Number of indirect call site descriptors: 0

RUN: llvm-objdump --disassemble-symbols=main %t.instrumented \
RUN:   | FileCheck %s --check-prefix=CHECK-MAIN

RUN: llvm-objdump --disassemble-symbols=foo %t.instrumented \
RUN:   | FileCheck %s --check-prefix=CHECK-FOO

RUN: llvm-objdump --disassemble-symbols=__bolt_instr_ind_call_handler \
RUN:   %t.instrumented | FileCheck %s --check-prefix=CHECK-INSTR-INDIR-CALL
RUN: llvm-objdump --disassemble-symbols=__bolt_instr_ind_call_handler_func \
RUN:   %t.instrumented | FileCheck %s --check-prefix=CHECK-INSTR-INDIR-CALL-FUNC

CHECK-MAIN: mov w0, #0xa
CHECK-MAIN-NEXT: mov w1, #0x14
// store current values
CHECK-MAIN-NEXT: stp x0, x30, [sp
// load callsite id
CHECK-MAIN-NEXT: mov x0,
CHECK-MAIN-NEXT: stp x8, x0, [sp
CHECK-MAIN-NEXT: adrp x8,
CHECK-MAIN-NEXT: add x8, x8
// call instrumentation library handler function
CHECK-MAIN-NEXT: blr x8
// restore registers saved before
CHECK-MAIN-NEXT: ldr x8, [sp]
CHECK-MAIN-NEXT: ldp x0, x30, [sp]
// original indirect call instruction
CHECK-MAIN-NEXT: blr x8

// instrumented pattern for blr x0
CHECK-FOO: stp x1, x30, [sp
CHECK-FOO-NEXT: mov x1,
CHECK-FOO-NEXT: stp x0, x1, [sp
CHECK-FOO-NEXT: adrp x0
CHECK-FOO-NEXT: add x0, x0
CHECK-FOO-NEXT: blr x0
CHECK-FOO-NEXT: ldr x0, [sp]
CHECK-FOO-NEXT: ldp x1, x30, [sp]
CHECK-FOO-NEXT: blr x0

CHECK-INSTR-INDIR-CALL: __bolt_instr_ind_call_handler>:
CHECK-INSTR-INDIR-CALL-NEXT: ret

CHECK-INSTR-INDIR-CALL-FUNC: __bolt_instr_ind_call_handler_func>:
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: adrp x0
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: ldr x0
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: cmp x0, #0x0
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: b.eq{{.*}}__bolt_instr_ind_call_handler
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: str x30
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: blr x0
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: ldr x30
CHECK-INSTR-INDIR-CALL-FUNC-NEXT: b{{.*}}__bolt_instr_ind_call_handler

# Instrumented program needs to finish returning zero
RUN: %t.instrumented | FileCheck %s -check-prefix=CHECK-OUTPUT

# Test that the instrumented data makes sense
RUN:  llvm-bolt %t.exe -o %t.bolted --data %t.fdata \
RUN:    --reorder-blocks=ext-tsp --reorder-functions=hfsort+ \
RUN:    --print-only=main,foo --print-finalized | FileCheck %s

RUN: %t.bolted | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: The sum is: 30

# Check that our indirect call has 1 hit recorded in the fdata file and that
# this was processed correctly by BOLT
CHECK-LABEL: Binary Function "foo"
CHECK:         blr     x0 # CallProfile: 1 (0 misses) :
CHECK-NEXT:    { getConst: 1 (0 misses) }

CHECK-LABEL: Binary Function "main"
CHECK:         blr     x8 # CallProfile: 1 (0 misses) :
CHECK-NEXT:    { add: 1 (0 misses) }
*/
