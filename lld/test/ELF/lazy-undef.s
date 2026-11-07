# REQUIRES: x86
## foo is only referenced by a lazy object file that is never extracted. A worker
## thread may intern foo while parsing, but SymbolTable::find must not see it.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld --threads=4 a.o --start-lib b.o --end-lib -T lds -e foo -o out 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
# RUN: llvm-readelf -s out | FileCheck %s --implicit-check-not=foo --implicit-check-not=bar

# WARN: warning: cannot find entry symbol foo; not setting start address
# CHECK: 0000000000000002 0 NOTYPE GLOBAL DEFAULT ABS x

#--- a.s
.globl _start
_start:
  ret

#--- b.s
.globl bar
bar:
  call foo

#--- lds
SECTIONS {
  .text : { *(.text) }
  x = DEFINED(foo) ? 1 : 2;
}
