# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'foo { foo*; }; bar { *; };' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=FOO %s

# RUN: echo 'foo { foo*; }; bar { f*; };' > %t.ver
# RUN: ld.lld --version-script %t.ver %t.o -shared -o %t.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=BAR %s

# RUN: echo 'bar1 { *; }; bar2 { *; };' > %t2.ver
# RUN: ld.lld --version-script %t2.ver %t.o -shared -o %t2.so 2>&1 | \
# RUN:   FileCheck --check-prefix=DUPWARN %s
# RUN: llvm-readelf --dyn-syms %t2.so | FileCheck --check-prefix=BAR2 %s

## If both a non-* glob and a * match, non-* wins.
## This is GNU linkers' behavior. We don't feel strongly this should be supported.
# FOO: GLOBAL DEFAULT 7 foo@@foo

# BAR: GLOBAL DEFAULT 7 foo@@bar

## When there are multiple * patterns, the last wins.
# BAR2: GLOBAL DEFAULT 7 foo@@bar2
# DUPWARN: warning: wildcard pattern '*' is used for multiple version definitions in version script

.globl foo
foo:
