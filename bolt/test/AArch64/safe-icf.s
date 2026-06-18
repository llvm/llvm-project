# Test BOLT can do safe ICF for AArch64.

# REQUIRES: system-linux,asserts

# RUN: %clang %cflags -x assembler-with-cpp %s -o %t1.so -Wl,-q \
# RUN:    -Wl,-z,notext -DREF_BY_NON_CONTROL_FLOW_INSTR
# RUN: %clang %cflags -x assembler-with-cpp %s -o %t2.so -Wl,-q \
# RUN:    -Wl,-z,notext -DREF_IN_RW_DATA_SEC
# RUN: %clang %cflags -x assembler-with-cpp %s -o %t3.so -Wl,-q \
# RUN:    -Wl,-z,notext -DNO_DUMMY_TEXT_RELOC

# RUN: llvm-bolt %t1.so -o %t.bolt --no-threads --debug-only=bolt-icf \
# RUN:   --icf=all 2>&1 | FileCheck %s --check-prefix=ICF-ALL
# RUN: llvm-bolt %t2.so -o %t.bolt --no-threads --debug-only=bolt-icf \
# RUN:   --icf=all 2>&1 | FileCheck %s --check-prefix=ICF-ALL
# RUN: llvm-bolt %t3.so -o %t.bolt --no-threads --debug-only=bolt-icf \
# RUN:   --icf=all 2>&1 | FileCheck %s --check-prefix=ICF-ALL

# RUN: llvm-bolt %t1.so -o %t.bolt --no-threads --debug-only=bolt-icf \
# RUN:   --icf=safe 2>&1 | FileCheck %s --check-prefix=ICF-SAFE
# RUN: llvm-bolt %t2.so -o %t.bolt --no-threads --debug-only=bolt-icf \
# RUN:   --icf=safe 2>&1 | FileCheck %s --check-prefix=ICF-SAFE

# RUN: not llvm-bolt %t3.so -o %t.bolt --icf=safe 2>&1 | FileCheck %s \
# RUN:   --check-prefix=ERROR

# ICF-ALL:  folding bar into foo
# ICF-SAFE: skipping function with reference taken foo
# ERROR:    binary built without relocations. Safe ICF is not supported

  .text

  .global foo
  .type foo, %function
foo:
  mov x0, #0x10
  ret

  .global bar
  .type bar, %function
bar:
  mov x0, #0x10
  ret

#if defined(REF_IN_RW_DATA_SEC) && !defined(NO_DUMMY_TEXT_RELOC)
  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
#endif

#if defined(REF_BY_NON_CONTROL_FLOW_INSTR)
  .global random
  .type random, %function
random:
  adrp x8, foo
  add  x8, x8, :lo12:foo
  br   x8
#endif

#if defined(REF_IN_RW_DATA_SEC)
  .data
  .balign 8
  .global funcptr
funcptr:
  .xword foo
#endif

  .section .rodata
  .global _ZTVxx
  .balign 8
_ZTVxx:
  .xword foo
  .xword bar
  .size _ZTVxx, .-_ZTVxx
