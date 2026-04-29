# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-as %t/bitcode.ll -o %t/bitcode.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/native.s -o %t/native.o

# RUN: %lld -dylib -lSystem -arch arm64 %t/native.o %t/bitcode.o -o %t/generated.dylib
# RUN: %lld -dylib -lSystem -arch arm64 --emit-tbd-only=%t/lld-generated.tbd %t/native.o %t/bitcode.o -o %t/generated.dylib

# RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4

# RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

#--- bitcode.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@weak_thread_local_var = weak thread_local global i33 0, align 4
@thread_local_var = thread_local global i32 0, align 4

#--- native.s

.section	__DATA,__thread_vars,thread_local_variables

.global _thread_local
_thread_local:
    ret

.global _thread_local_and_weak
_thread_local_and_weak:
    ret
