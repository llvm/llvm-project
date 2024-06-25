/**
 ** Test for asm-dump functionality.
 *
 * REQUIRES: x86_64-linux,bolt-runtime
 *
 * Compile the source
 * RUN: %clang -fPIC %s -o %t.exe -Wl,-q
 *
 * Profile collection: instrument the binary
 * RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata -o \
 * RUN:   %t.instr
 *
 * Profile collection: run instrumented binary (and capture output)
 * RUN: %t.instr > %t.result
 *
 * Run BOLT with asm-dump
 * RUN: llvm-bolt %t.exe -p %t.fdata --funcs=main --asm-dump=%t -o %t.null \
 * RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
 *
 * Check asm file contents
 * RUN: cat %t/main.s | FileCheck %s --check-prefix=CHECK-FILE
 *
 * Now check if asm-dump file can be consumed by BOLT infra
 * Strip dot from compiler-local symbols:
 * RUN: sed -i 's/\.L/L/g' %t/main.s
 *
 * Recompile the asm file into objfile
 * RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/main.s -o %t.o
 *
 * Reconstruct fdata
 * RUN: link_fdata %t/main.s %t.o %t.fdata.reconst
 *
 * XXX: reenable once dumping data is supported
 * Check if reoptimized file produces the same results
 * dontrun: %t.exe.reopt > %t.result.reopt
 * dontrun: cmp %t.result %t.result.reopt
 *
 * Delete our BB symbols so BOLT doesn't mark them as entry points
 * RUN: llvm-strip --strip-unneeded %t.o
 *
 * Recompile the binary
 * RUN: %clang -fPIC %t.o -o %t.exe.reopt -Wl,-q
 *
 * Finally consume reoptimized file with reconstructed fdata
 * RUN: llvm-bolt %t.exe.reopt -p %t.fdata.reconst -o %t.null \
 * RUN:   | FileCheck %s --check-prefix=CHECK-REOPT
 *
 * CHECK-BOLT: BOLT-INFO: Dumping function assembly to {{.*}}/main.s
 *
 * CHECK-FILE:      .globl main
 * CHECK-FILE-NEXT: .type main, %function
 * CHECK-FILE-NEXT: main:
 * CHECK-FILE-NEXT: # FDATA: 0 [unknown] 0 1 main 0 0 1
 * CHECK-FILE-NEXT: .cfi_startproc
 * CHECK-FILE-NEXT: .LBB{{.*}}:
 * CHECK-FILE:      .cfi_def_cfa_offset 16
 * CHECK-FILE:      leaq  {{.*}}(%rip)
 * CHECK-FILE:      callq puts@PLT
 * CHECK-FILE:      .cfi_endproc
 * CHECK-FILE-NEXT: .size main, .-main
 * CHECK-FILE:      .section .rodata
 *
 * CHECK-REOPT: BOLT-INFO: 1 out of {{.*}} functions in the binary {{.*}} have
 * CHECK-REOPT: non-empty execution profile
 */
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
  for (int I = 0; I < 10; I++) {
    if (I != 9)
      continue;
    if (argc > 1 &&
        strncmp(argv[1], "--help", strlen("--help")) == 0) {
      puts("Help message\n");
    } else {
      puts("Hello, World!\n");
    }
  }
  return 0;
}
