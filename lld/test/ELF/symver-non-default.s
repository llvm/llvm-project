# REQUIRES: x86
## Test symbol resolution related to .symver produced non-default version symbols.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ref.s -o %t/ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def1.s -o %t/def1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def2.s -o %t/def2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/def3.s -o %t/def3.o

## foo@v1 & foo defined at the same location are combined.
# RUN: ld.lld -shared --version-script=%t/ver1 %t/def1.o %t/ref.o -o %t1
# RUN: llvm-readelf --dyn-syms %t1 | FileCheck %s --check-prefix=CHECK1
# RUN: ld.lld -shared --version-script=%t/ver2 %t/def1.o %t/ref.o -o %t1
# RUN: llvm-readelf --dyn-syms %t1 | FileCheck %s --check-prefix=CHECK1

# CHECK1:       1: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@v1
# CHECK1-EMPTY:

## def2.o doesn't define foo. foo@v1 & undefined foo are unrelated.
# RUN: ld.lld -shared --version-script=%t/ver1 %t/def2.o %t/ref.o -o %t2
# RUN: llvm-readelf -r --dyn-syms %t2 | FileCheck %s --check-prefix=CHECK2

# CHECK2:       R_X86_64_JUMP_SLOT {{.*}} foo + 0
# CHECK2:       1: {{.*}} NOTYPE GLOBAL DEFAULT UND   foo{{$}}
# CHECK2-NEXT:  2: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@v1
# CHECK2-EMPTY:

## def2.o doesn't define foo. foo@v1 & defined foo are unrelated.
# RUN: ld.lld -shared --version-script=%t/ver1 %t/def2.o %t/def3.o %t/ref.o -o %t3
# RUN: llvm-readelf -r --dyn-syms %t3 | FileCheck %s --check-prefix=CHECK3

# CHECK3:       R_X86_64_JUMP_SLOT {{.*}} foo + 0
# CHECK3:       1: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@v1
# CHECK3-NEXT:  2: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo{{$}}
# CHECK3-EMPTY:

## foo@v1 overrides the defined foo which is affected by a version script.
# RUN: ld.lld -shared --version-script=%t/ver2 %t/def2.o %t/def3.o %t/ref.o -o %t4
# RUN: llvm-readelf -r --dyn-syms %t4 | FileCheck %s --check-prefix=CHECK4

# CHECK4:       R_X86_64_JUMP_SLOT {{.*}} foo@v1 + 0
# CHECK4:       1: {{.*}} NOTYPE GLOBAL DEFAULT [[#]] foo@v1
# CHECK4-EMPTY:

#--- ver1
v1 {};

#--- ver2
v1 { foo; };

#--- ref.s
call foo

#--- def1.s
.globl foo
.symver foo, foo@v1
foo:
  ret

#--- def2.s
.globl foo_v1
.symver foo_v1, foo@v1, remove
foo_v1:
  ret

#--- def3.s
.globl foo
foo:
  ret
