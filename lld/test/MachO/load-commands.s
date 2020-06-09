# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -arch x86_64 -o %t %t.o

## Check for the presence of load commands that are essential for a working
## executable.
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s
# CHECK-DAG: cmd LC_DYLD_INFO_ONLY
# CHECK-DAG: cmd LC_SYMTAB
# CHECK-DAG: cmd LC_DYSYMTAB
# CHECK-DAG: cmd LC_MAIN
# CHECK-DAG: cmd LC_LOAD_DYLINKER

## Check for the absence of load commands that should not be in an executable.
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s --check-prefix=NCHECK
# NCHECK-NOT: cmd: LC_ID_DYLIB

.text
.global _main
_main:
  mov $0, %rax
  ret
