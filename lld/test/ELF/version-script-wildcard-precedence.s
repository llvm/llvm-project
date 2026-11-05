# REQUIRES: x86
## Test precedence among version script wildcards.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld --version-script later.ver a.o -shared -o later.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms later.so | FileCheck %s -DV=bar
# RUN: ld.lld --version-script node.ver a.o -shared -o node.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms node.so | FileCheck %s -DV=v1
# RUN: ld.lld --version-script local.ver a.o -shared -o local.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms local.so | FileCheck %s -DV=v2

# CHECK:      UND
# CHECK-NEXT: [[#]] foo1@@[[V]]
# CHECK-NEXT: [[#]] foo2@@[[V]]
# CHECK-NEXT: [[#]] bar{{$}}
# CHECK-NEXT: [[#]] _Z3fooi{{$}}
# CHECK-NOT:  {{.}}

## If both a non-* pattern and * match, non-* wins even if * comes later.
## This is GNU linkers' behavior. We don't feel strongly this should be supported.
# RUN: ld.lld --version-script star.ver a.o -shared -o star.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms star.so | FileCheck %s --check-prefix=STAR
# STAR:      UND
# STAR-NEXT: [[#]] foo1@@foo
# STAR-NEXT: [[#]] foo2@@foo
# STAR-NEXT: [[#]] bar@@bar
# STAR-NEXT: [[#]] _Z3fooi@@bar
# STAR-NOT:  {{.}}

## When there are multiple * patterns, the last wins.
# RUN: ld.lld --version-script star2.ver a.o -shared -o star2.so 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DUPWARN
# RUN: llvm-readelf --dyn-syms star2.so | FileCheck %s --check-prefix=STAR2
# DUPWARN: warning: wildcard pattern '*' is used for multiple version definitions in version script
# STAR2:      UND
# STAR2-NEXT: [[#]] foo1@@bar2
# STAR2-NEXT: [[#]] foo2@@bar2
# STAR2-NEXT: [[#]] bar@@bar2
# STAR2-NEXT: [[#]] _Z3fooi@@bar2
# STAR2-NOT:  {{.}}

## An exact name beats a wildcard in a later definition.
# RUN: ld.lld --version-script exact.ver a.o -shared -o exact.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms exact.so | FileCheck %s --check-prefix=EXACT
# EXACT:      UND
# EXACT-NEXT: [[#]] foo1@@v1
# EXACT-NEXT: [[#]] foo2@@v2
# EXACT-NOT:  {{.}}

## An extern "C++" wildcard matches demangled names, which for a non-mangled
## symbol is the name itself, and takes part in the same precedence.
# RUN: ld.lld --version-script cxx.ver a.o -shared -o cxx.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms cxx.so | FileCheck %s --check-prefix=CXX
# CXX:      UND
# CXX-NEXT: [[#]] foo1@@v2
# CXX-NEXT: [[#]] foo2@@v2
# CXX-NEXT: [[#]] bar{{$}}
# CXX-NEXT: [[#]] _Z3fooi@@v2
# CXX-NOT:  {{.}}

#--- a.s
.globl foo1, foo2, bar, _Z3fooi
foo1:
foo2:
bar:
_Z3fooi:

#--- later.ver
foo { foo*; }; bar { f*; };

#--- node.ver
v1 { global: foo*; local: f*; };

#--- local.ver
v1 { local: f*; }; v2 { global: foo*; };

#--- star.ver
foo { foo*; }; bar { *; };

#--- star2.ver
bar1 { *; }; bar2 { *; };

#--- exact.ver
v1 { global: foo1; local: *; }; v2 { global: foo*; };

#--- cxx.ver
v1 { global: foo*; }; v2 { global: extern "C++" { foo*; }; };
