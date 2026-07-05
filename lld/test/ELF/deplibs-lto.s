# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 -o deplibs.o deplibs.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 -o foo.o foo.s
# RUN: llvm-as -o lto.o lto.ll
# RUN: llvm-ar rc libdeplibs.a deplibs.o
# RUN: llvm-ar rc libfoo.a foo.o

## LTO emits a libcall (__aarch64_ldadd4_relax) that is resolved using a
## library (libdeplibs.a) that contains a .deplibs section pointing to a file
## (libfoo.a) not yet added to the link.
# RUN: not ld.lld lto.o -u a -L. -ldeplibs 2>&1 | FileCheck %s
# CHECK: error: input file 'foo.o' added after LTO

## Including the file before LTO prevents the issue.
# RUN: ld.lld lto.o -u a -L. -ldeplibs -lfoo

#--- foo.s
.global foo
foo:
#--- deplibs.s
.global __aarch64_ldadd4_relax
__aarch64_ldadd4_relax:
    b foo
.section ".deplibs","MS",@llvm_dependent_libraries,1
    .asciz "foo"
#--- lto.ll
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define void @a(i32* nocapture %0) #0 {
  %2 = atomicrmw add i32* %0, i32 1 monotonic, align 4
  ret void
}

attributes #0 = { "target-features"="+outline-atomics" }
