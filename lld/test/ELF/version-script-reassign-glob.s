# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'foo { foo*; }; bar { *; };' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=FOO %s

# RUN: echo 'foo { foo*; }; bar { f*; };' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=BAR %s

## If both a non-* glob and a * match, non-* wins.
## This is GNU linkers' behavior. We don't feel strongly this should be supported.
# FOO: GLOBAL DEFAULT 7 foo@@foo

# BAR: GLOBAL DEFAULT 7 foo@@bar

.globl foo
foo:
