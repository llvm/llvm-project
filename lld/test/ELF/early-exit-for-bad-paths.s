# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: not ld.lld %t.o -o does_not_exist/output 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=NO-DIR,CHECK
# RUN: not ld.lld %t.o -o %s/dir_is_a_file 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=DIR-IS-FILE,CHECK

# RUN: echo "OUTPUT(\"does_not_exist/output\")" > %t.script
# RUN: not ld.lld %t.o %t.script 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=NO-DIR,CHECK
# RUN: echo "OUTPUT(\"%s/dir_is_a_file\")" > %t.script
# RUN: not ld.lld %t.o %t.script 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=DIR-IS-FILE,CHECK

# NO-DIR: error: cannot open output file does_not_exist/output:
# DIR-IS-FILE: error: cannot open output file {{.*}}/dir_is_a_file:

# We should exit before doing the actual link. If an undefined symbol error is
# discovered we haven't bailed out early as expected.
# CHECK-NOT: undefined_symbol

# RUN: not ld.lld %t.o -o / 2>&1 | FileCheck %s -check-prefixes=ROOT
# ROOT: error: cannot open output file /

  .globl _start
_start:
  call undefined_symbol
