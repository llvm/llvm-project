#include <stdio.h>

typedef int (*func_ptr)(int, int);

int add(int a, int b) { return a + b; }

int main() {
  func_ptr fun;
  fun = add;
  int sum = fun(10, 20); // indirect call to 'add'
  printf("The sum is: %d\n", sum);
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
RUN:   | FileCheck %s --check-prefix=CHECK-INSTR-LOG

CHECK-INSTR-LOG: BOLT-INSTRUMENTER: Number of indirect call site descriptors: 1

RUN: llvm-objdump --disassemble-symbols=main %t.instrumented \
RUN:   | FileCheck %s --check-prefix=CHECK-INSTR-INDIRECTREG

RUN: llvm-objdump --disassemble-symbols=__bolt_instr_ind_call_handler \
RUN:   %t.instrumented | FileCheck %s --check-prefix=CHECK-INSTR-INDIR-CALL
RUN: llvm-objdump --disassemble-symbols=__bolt_instr_ind_call_handler_func \
RUN:   %t.instrumented | FileCheck %s --check-prefix=CHECK-INSTR-INDIR-CALL-FUNC

CHECK-INSTR-INDIRECTREG: mov w0, #0xa
CHECK-INSTR-INDIRECTREG-NEXT: mov w1, #0x14
// store current values
CHECK-INSTR-INDIRECTREG-NEXT: stp x0, x1, {{.*}}
// store the indirect target address in x0
CHECK-INSTR-INDIRECTREG-NEXT: mov x0, x8
// load callsite id into x1
CHECK-INSTR-INDIRECTREG-NEXT: movk x1, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: movk x1, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: movk x1, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: movk x1, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: stp x0, x30, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: adrp x8, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: add x8, {{.*}}
// call instrumentation library handler function
CHECK-INSTR-INDIRECTREG-NEXT: blr x8
// restore registers saved before
CHECK-INSTR-INDIRECTREG-NEXT: ldp x0, x30, {{.*}}
CHECK-INSTR-INDIRECTREG-NEXT: mov x8, x0
CHECK-INSTR-INDIRECTREG-NEXT: ldp x0, x1, {{.*}}
// original indirect call instruction
CHECK-INSTR-INDIRECTREG-NEXT: blr x8


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
RUN:    --print-only=main --print-finalized | FileCheck %s

RUN: %t.bolted | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: The sum is: 30

# Check that our indirect call has 1 hit recorded in the fdata file and that
# this was processed correctly by BOLT
CHECK:         blr     x8 # CallProfile: 1 (0 misses) :
CHECK-NEXT:    { add: 1 (0 misses) }
*/
