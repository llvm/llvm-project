# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libgoodbye.s -o %t/goodbye.o
# RUN: llvm-ar --format=darwin crs %t/libgoodbye.a %t/goodbye.o
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test -L%t -lgoodbye %t/test.o
#
# RUN: llvm-objdump --syms -d -r %t/test | FileCheck %s
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 \
# RUN:   -static -no_pie -o %t/static -L%t -lgoodbye %t/test.o
# RUN: llvm-objdump --macho --private-header %t/static | FileCheck %s --check-prefix=STATIC-HEADER
# RUN: llvm-objdump --macho --all-headers %t/static | FileCheck %s --check-prefix=STATIC-LOADS

# CHECK: SYMBOL TABLE:
# CHECK: {{0+}}[[ADDR:[0-9a-f]+]] g     O __TEXT,__cstring _goodbye_world

# CHECK: Disassembly of section __TEXT,__text
# CHECK-LABEL: <_main>:
# CHECK: leaq {{.*}}(%rip), %rsi   ## 0x[[ADDR]] <_goodbye_world>

# STATIC-HEADER:      magic        cputype cpusubtype  caps    filetype {{.*}} flags
# STATIC-HEADER-NEXT: MH_MAGIC_64  X86_64         ALL  {{.*}}  EXECUTE  {{.*}} NOUNDEFS{{$}}

# STATIC-LOADS:      cmd LC_SYMTAB
# STATIC-LOADS-NOT:  cmd LC_DYLD_INFO_ONLY
# STATIC-LOADS-NOT:  cmd LC_DYLD_CHAINED_FIXUPS
# STATIC-LOADS-NOT:  cmd LC_DYLD_EXPORTS_TRIE
# STATIC-LOADS-NOT:  cmd LC_LOAD_DYLINKER
# STATIC-LOADS-NOT:  cmd LC_CODE_SIGNATURE

.section __TEXT,__text
.global _main

_main:
  movl $0x2000004, %eax                 # write()
  mov $1, %rdi                          # stdout
  leaq _goodbye_world(%rip), %rsi
  mov $15, %rdx                         # length
  syscall
  mov $0, %rax
  ret
