# This test is to validate instrumentation code sequence generated with
# and without `--no-lse-atomics`.

# REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

# RUN: %clang %cflags -pie %s -o %t.so -Wl,-q -Wl,--init=_foo -Wl,--fini=_foo

  .text
  .global _foo
  .type _foo, %function
_foo:
  ret

  .global _start
  .type _start, %function
_start:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE

# RUN: llvm-bolt %t.so -o %t.instr.so --instrument
# RUN: llvm-objdump -d %t.instr.so | FileCheck %s --check-prefix=INLINE
# INLINE: {{.*}} <_foo>:
# INLINE-NEXT: {{.*}} stp x0, x1, [sp, #-0x10]!
# INLINE-NEXT: {{.*}} adrp  x0, 0x{{[0-9a-f]*}} {{.*}}
# INLINE-NEXT: {{.*}} add x0, x0, #0x{{[0-9a-f]*}}
# INLINE-NEXT: {{.*}} mov x1, #0x1 
# INLINE-NEXT: {{.*}} stadd x1, [x0]
# INLINE-NEXT: {{.*}} ldp x0, x1, [sp], #0x10

# RUN: llvm-bolt %t.so -o %t.instr.no_lse.so --instrument \
# RUN:   --no-lse-atomics
# RUN: llvm-objdump -d %t.instr.no_lse.so | FileCheck %s --check-prefix=NOLSE
# NOLSE: {{.*}} <_foo>:
# NOLSE-NEXT: {{.*}} stp x0, x30, [sp, #-0x10]!
# NOLSE-NEXT: {{.*}} stp x1, x2, [sp, #-0x10]!
# NOLSE-NEXT: {{.*}} adrp x0, 0x{{[0-9a-f]*}} {{.*}}
# NOLSE-NEXT: {{.*}} add x0, x0, #0x{{[0-9a-f]*}}
# NOLSE-NEXT: {{.*}} adrp x1, 0x[[PAGEBASE:[0-9a-f]*]]000 {{.*}}
# NOLSE-NEXT: {{.*}} add x1, x1, #0x[[PAGEOFF:[0-9a-f]*]]
# NOLSE-NEXT: {{.*}} blr x1
# NOLSE-NEXT: {{.*}} ldp x0, x30, [sp], #0x10
# NOLSE: {{[0]*}}[[PAGEBASE]][[PAGEOFF]] <__bolt_instr_counter_incr>:
# NOLSE-NEXT: {{.*}} ldaxr x1, [x0]
# NOLSE-NEXT: {{.*}} add x1, x1, #0x1
# NOLSE-NEXT: {{.*}} stlxr w2, x1, [x0]
# NOLSE-NEXT: {{.*}} cbnz w2, 0x{{[0-9[a-f]*}} <__bolt_instr_counter_incr>
# NOLSE-NEXT: {{.*}} ldp x1, x2, [sp], #0x10
# NOLSE-NEXT: {{.*}} ret
