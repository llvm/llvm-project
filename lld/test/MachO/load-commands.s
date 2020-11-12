# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/executable %t/test.o
# RUN: %lld -bundle -o %t/bundle %t/test.o
# RUN: %lld -dylib -o %t/dylib %t/test.o

## These load commands should be in every final output binary.
# COMMON-DAG: cmd LC_DYLD_INFO_ONLY
# COMMON-DAG: cmd LC_SYMTAB
# COMMON-DAG: cmd LC_DYSYMTAB
# COMMON-DAG: cmd LC_UUID

## Check for the presence of load commands that are essential for a working
## executable. Also check that it has the right filetype.
# RUN: llvm-objdump --macho --all-headers %t/executable | FileCheck %s --check-prefix=COMMON
# RUN: llvm-objdump --macho --all-headers %t/executable | FileCheck %s --check-prefix=EXEC
# EXEC:      magic        cputype cpusubtype  caps    filetype
# EXEC-NEXT: MH_MAGIC_64  X86_64         ALL  {{.*}}  EXECUTE
# EXEC-DAG:  cmd LC_MAIN
# EXEC-DAG:  cmd LC_LOAD_DYLINKER

## Check for the absence of load commands that should not be in an executable.
# RUN: llvm-objdump --macho --all-headers %t/executable | FileCheck %s --check-prefix=NEXEC
# NEXEC-NOT: cmd: LC_ID_DYLIB

## Check for the presence / absence of load commands for the dylib.
# RUN: llvm-objdump --macho --all-headers %t/dylib | FileCheck %s --check-prefix=COMMON
# RUN: llvm-objdump --macho --all-headers %t/dylib | FileCheck %s --check-prefix=DYLIB
# DYLIB:      magic        cputype cpusubtype  caps    filetype
# DYLIB-NEXT: MH_MAGIC_64  X86_64         ALL  {{.*}}  DYLIB
# DYLIB:      cmd LC_ID_DYLIB

# RUN: llvm-objdump --macho --all-headers %t/bundle | FileCheck %s --check-prefix=NDYLIB
# NDYLIB-NOT: cmd: LC_MAIN
# NDYLIB-NOT: cmd: LC_LOAD_DYLINKER

## Check for the presence / absence of load commands for the bundle.
# RUN: llvm-objdump --macho --all-headers %t/bundle | FileCheck %s --check-prefix=COMMON
# RUN: llvm-objdump --macho --all-headers %t/bundle | FileCheck %s --check-prefix=BUNDLE
# BUNDLE:      magic        cputype cpusubtype  caps    filetype
# BUNDLE-NEXT: MH_MAGIC_64  X86_64         ALL  {{.*}}  BUNDLE

# RUN: llvm-objdump --macho --all-headers %t/bundle | FileCheck %s --check-prefix=NBUNDLE
# NBUNDLE-NOT: cmd: LC_MAIN
# NBUNDLE-NOT: cmd: LC_LOAD_DYLINKER

.text
.global _main
_main:
  mov $0, %rax
  ret
