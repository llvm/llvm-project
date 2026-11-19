# REQUIRES: x86

## A symbol defined by --defsym takes precedence over a PROVIDE directive and
## its value is used in linker script expressions, regardless of whether -T
## or --defsym comes first on the command line (GNU ld compatible).

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux a.s -o a.o
# RUN: ld.lld -o t1 a.o -T script.ld --defsym=_a=20
# RUN: ld.lld -o t2 a.o --defsym=_a=20 -T script.ld
# RUN: ld.lld -o t3 a.o -T script.ld
# RUN: llvm-readelf -s t1 | FileCheck %s --check-prefix=DEFSYM
# RUN: llvm-readelf -s t2 | FileCheck %s --check-prefix=DEFSYM
# RUN: llvm-readelf -W -S t1 | FileCheck %s --check-prefix=DEFSYM-SIZE
# RUN: llvm-readelf -W -S t2 | FileCheck %s --check-prefix=DEFSYM-SIZE
# RUN: llvm-readelf -W -S t3 | FileCheck %s --check-prefix=PROVIDE-SIZE

# DEFSYM: 0000000000000014     0 NOTYPE  GLOBAL DEFAULT   ABS _a

## _a * 1024 = 20 * 1024 = 0x5000.
# DEFSYM-SIZE: .test {{.*}} 005000
## _a * 1024 = 10 * 1024 = 0x2800.
# PROVIDE-SIZE: .test {{.*}} 002800

#--- a.s
.text
.globl _start
_start:
  ret

#--- script.ld
PROVIDE(_a = 10);
SECTIONS
{
  .test :
  {
    . += _a * 1024;
  }
}
