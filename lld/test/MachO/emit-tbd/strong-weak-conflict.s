# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-as %t/bitcode_weak.ll -o %t/bitcode_weak.o
# RUN: llvm-as %t/bitcode_strong.ll -o %t/bitcode_strong.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/native_weak.s -o %t/native_weak.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/native_strong.s -o %t/native_strong.o

# RUN: echo %t/bitcode_weak.o > %t/filelist.txt
# RUN: echo %t/bitcode_strong.o >> %t/filelist.txt
# RUN: echo %t/native_strong.o >> %t/filelist.txt
# RUN: echo %t/native_weak.o >> %t/filelist.txt

# RUN: echo %t/native_weak.o > %t/filelist-reversed.txt
# RUN: echo %t/native_strong.o >> %t/filelist-reversed.txt
# RUN: echo %t/bitcode_strong.o >> %t/filelist-reversed.txt
# RUN: echo %t/bitcode_weak.o >> %t/filelist-reversed.txt

# RUN: %lld -dylib -lSystem -arch arm64 -filelist %t/filelist.txt -o %t/generated.dylib
# RUN: %lld -dylib -lSystem -arch arm64 --emit-tbd-only=%t/lld-generated.tbd -filelist %t/filelist.txt -o %t/generated.dylib
# RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4
# RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

# RUN: %lld -dylib -lSystem -arch arm64 -filelist %t/filelist-reversed.txt -o %t/generated-reversed.dylib
# RUN: %lld -dylib -lSystem -arch arm64 --emit-tbd-only=%t/lld-generated-reversed.tbd -filelist %t/filelist-reversed.txt -o %t/generated-reversed.dylib
# RUN: llvm-readtapi -stubify %t/generated-reversed.dylib -o %t/tapi-generated-reversed.tbd --filetype=tbd-v4
# RUN: llvm-readtapi -compare %t/lld-generated-reversed.tbd %t/tapi-generated-reversed.tbd

#--- bitcode_weak.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@bar = weak global i32 0, align 4

#--- bitcode_strong.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@bar = global i32 0, align 4

#--- native_weak.s

.global _foo
.weak_definition _foo
_foo:
  ret

#--- native_strong.s

.global _foo
_foo:
  ret
