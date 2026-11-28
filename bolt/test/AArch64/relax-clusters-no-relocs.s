# Check automatic and explicit relaxation choices without code relocations.

# RUN: %clang %cflags %s -o %t.exe

# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 \
# RUN:   | FileCheck %s --check-prefix=LEGACY

# RUN: llvm-bolt %t.exe -o %t.bolt --compact-code-model 2>&1 \
# RUN:   | FileCheck %s --check-prefix=COMPACT

# RUN: not llvm-bolt %t.exe -o %t.bolt --relax-with-clusters 2>&1 \
# RUN:   | FileCheck %s --check-prefix=ERROR

# LEGACY: BOLT-WARNING: clustered relaxation requires code relocations; falling back to the stub-insertion pass
# LEGACY: BOLT-INFO: Starting stub-insertion pass

# COMPACT: BOLT-WARNING: clustered relaxation requires code relocations; falling back to local branch relaxation
# COMPACT: BOLT-INFO: relaxing branches for compact code model (<128MB)

# ERROR: BOLT-ERROR: clustered relaxation requires code relocations; link with --emit-relocs or pass --relax-with-clusters=false

  .text
  .globl _start
  .type _start, %function
_start:
  ret
  .size _start, .-_start
