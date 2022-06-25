# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "TARGET(binary) INPUT(\"%t.o\") TARGET(elf64-x86-64) INPUT(\"%t.o\")" > %t.script
# RUN: ld.lld --script %t.script -o %t.exe
# RUN: llvm-readelf -s %t.exe | FileCheck %s
# RUN: echo 'TARGET("elf64-x86-64") INPUT("%t.o") TARGET("binary") INPUT("%t.o")' > %t.script
# RUN: ld.lld -T %t.script -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK-DAG: [[#]] _binary_
# CHECK-DAG: [[#]] foobar

# RUN: echo "TARGET(foo)" > %t2.script
# RUN: not ld.lld --script %t2.script -o /dev/null 2>&1 | FileCheck -check-prefix=ERR %s

# ERR: unknown target: foo

.global foobar
foobar:
  nop
