# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: echo ".globl foo; foo:" | llvm-mc -filetype=obj -triple=x86_64-unknown-linux - -o %t2.o
# RUN: rm -f %t2.a
# RUN: llvm-ar rcs %t2.a %t2.o

# RUN: ld.lld --fatal-warnings -o %t.exe %t1.o %t2.a
# RUN: ld.lld --fatal-warnings -o %t.exe %t2.a %t1.o
# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t1.o %t2.a
# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t1.o --start-lib %t2.o --end-lib

# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe --start-group %t2.a %t1.o --end-group
# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe "-(" %t2.a %t1.o "-)"

# RUN: echo "INPUT(\"%t1.o\" \"%t2.a\")" > %t1.script
# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t1.script

# RUN: echo "GROUP(\"%t2.a\" \"%t1.o\")" > %t2.script
# RUN: ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t2.script

# RUN: not ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t2.a %t1.o 2>&1 | FileCheck %s
# RUN: not ld.lld --fatal-warnings --warn-backrefs -o %t.exe %t2.a "-(" %t1.o "-)" 2>&1 | FileCheck %s

# CHECK: backward reference detected: foo in {{.*}}1.o refers to {{.*}}2.a

# RUN: not ld.lld --fatal-warnings --start-group --start-group 2>&1 | FileCheck -check-prefix=START %s
# START: nested --start-group

# RUN: not ld.lld --fatal-warnings --end-group 2>&1 | FileCheck -check-prefix=END %s
# END: stray --end-group

.globl _start, foo
_start:
  call foo
