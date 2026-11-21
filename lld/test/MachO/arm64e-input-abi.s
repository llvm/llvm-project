# REQUIRES: aarch64
## The CPU subtype of an arm64e object also holds its pointer authentication ABI
## version. Like ld-prime, an arm64e link takes only objects with userland
## version 0, even on macOS, whose executables are marked version 1. Static and
## preload links also take unversioned and kernel objects. Any other object is
## ignored with a warning, or rejected with -arch_errors_fatal, and named the
## way ld-prime names it.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos11.0 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/main.s -o %t/main-arm64.o
# RUN: yaml2obj -DSUBTYPE=0x80000002 %t/helper.yaml -o %t/v0.o
# RUN: yaml2obj -DSUBTYPE=0x2 %t/helper.yaml -o %t/old.o
# RUN: yaml2obj -DSUBTYPE=0x81000002 %t/helper.yaml -o %t/v1.o
# RUN: yaml2obj -DSUBTYPE=0xC0000002 %t/helper.yaml -o %t/kernel.o
# RUN: yaml2obj -DSUBTYPE=0xC1000002 %t/helper.yaml -o %t/kernel-v1.o
# RUN: yaml2obj -DSUBTYPE=0x82000002 %t/helper.yaml -o %t/v2.o
# RUN: llvm-ar rcs %t/v1.a %t/v1.o

# DEFINE: %{lld} = %no-fatal-warnings-lld -arch arm64e -arch_errors_fatal \
# DEFINE:   %t/main.o -o %t/out

# RUN: %{lld} %t/v0.o
# RUN: not %{lld} %t/old.o 2>&1 | FileCheck %s --check-prefix=OLD
# RUN: not %{lld} %t/v1.o 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %{lld} %t/v1.a 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %{lld} %t/kernel.o 2>&1 | FileCheck %s --check-prefix=KERNEL
# RUN: not %{lld} %t/kernel-v1.o 2>&1 | FileCheck %s --check-prefix=KERNEL-V1
# RUN: not %{lld} %t/v2.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN

# RUN: %{lld} -static %t/old.o
# RUN: %{lld} -static %t/kernel.o
# RUN: %{lld} -preload %t/old.o
# RUN: %{lld} -preload %t/kernel.o
# RUN: not %{lld} -static %t/v1.o 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %{lld} -static %t/kernel-v1.o 2>&1 | FileCheck %s --check-prefix=KERNEL-V1
# RUN: not %{lld} -preload %t/kernel-v1.o 2>&1 | FileCheck %s --check-prefix=KERNEL-V1

# RUN: %no-fatal-warnings-lld -arch arm64e %t/main.o %t/v1.o -o %t/out 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARN
# RUN: not %no-fatal-warnings-lld -arch arm64 -arch_errors_fatal %t/main-arm64.o \
# RUN:   %t/v1.o -o %t/out 2>&1 | FileCheck %s --check-prefix=V1-ARM64

# OLD:       error: {{.*}}old.o has architecture arm64e.old which is incompatible with target architecture arm64e
# V1:        error: {{.*}}v1.{{[oa]}} has architecture arm64e.v1 which is incompatible with target architecture arm64e
# KERNEL:    error: {{.*}}kernel.o has architecture arm64e.kernel which is incompatible with target architecture arm64e
# KERNEL-V1: error: {{.*}}kernel-v1.o has architecture arm64e.kernel.v1 which is incompatible with target architecture arm64e
# UNKNOWN:   error: {{.*}}v2.o has architecture unknown which is incompatible with target architecture arm64e
# WARN:      warning: {{.*}}v1.o has architecture arm64e.v1 which is incompatible with target architecture arm64e
# V1-ARM64:  error: {{.*}}v1.o has architecture arm64e.v1 which is incompatible with target architecture arm64

#--- main.s
.globl _main
.p2align 2
_main:
  ret

#--- helper.yaml
## An object with one symbol, so that an archive of it has a symbol table.
--- !mach-o
IsLittleEndian:  true
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      [[SUBTYPE]]
  filetype:        0x1
  ncmds:           1
  sizeofcmds:      24
  flags:           0x0
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          56
    nsyms:           1
    stroff:          72
    strsize:         9
LinkEditData:
  NameList:
    - n_strx:          1
      n_type:          0x3
      n_sect:          0
      n_desc:          0
      n_value:         0
  StringTable:
    - ''
    - _helper
...
