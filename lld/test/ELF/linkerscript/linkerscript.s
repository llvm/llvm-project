# There is some bad quoting interaction between lit's internal shell, which is
# implemented in Python, and the Cygwin implementations of the Unix utilities.
# Avoid running these tests on Windows for now by requiring a real shell.

# REQUIRES: shell

# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/libsearch-st.s -o %t2.o

# RUN: echo "EXTERN( undef undef2 )" > %t.script
# RUN: ld.lld %t -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "OUTPUT_FORMAT(elf64-x86-64) /*/*/ GROUP(\"%t\" )" > %t.script
# RUN: ld.lld -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: rm -f %t.out
# RUN: echo "OUTPUT(\"%t.out\")" > %t.script
# RUN: ld.lld %t.script %t
# RUN: llvm-readobj %t.out > /dev/null

# RUN: echo "SEARCH_DIR(/lib/foo/blah)" > %t.script
# RUN: ld.lld %t.script %t
# RUN: llvm-readobj %t.out > /dev/null

# RUN: echo ";SEARCH_DIR(x);SEARCH_DIR(y);" > %t.script
# RUN: ld.lld %t.script %t
# RUN: llvm-readobj %t.out > /dev/null

# RUN: echo ";" > %t.script
# RUN: ld.lld %t.script %t
# RUN: llvm-readobj %t.out > /dev/null

# RUN: echo "INCLUDE \"%t.script2\" OUTPUT(\"%t.out\")" > %t.script1
# RUN: echo "GROUP(\"%t\")" > %t.script2
# RUN: ld.lld %t.script1
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "FOO(BAR)" > %t.script
# RUN: not ld.lld -o foo %t.script > %t.log 2>&1
# RUN: FileCheck -check-prefix=ERR1 %s < %t.log

# ERR1: unknown directive: FOO

.globl _start, _label
_start:
  ret
_label:
  ret
