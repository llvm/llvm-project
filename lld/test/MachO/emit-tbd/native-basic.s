# RUN: rm -rf %t
# RUN: split-file %s %t

# (1) Produce the object files
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios-simulator15.1.0 %t/baz.s -o %t/baz.o

# (2) Produce a tbd and a dylib with lld
# RUN: %lld -dylib -arch arm64 %t/foo.o %t/bar.o %t/baz.o -o %t/generated.dylib
# RUN: %lld -dylib -arch arm64 --emit-tbd-only=%t/lld-generated.tbd %t/foo.o %t/bar.o %t/baz.o -o %t/generated.dylib

# (3) Produce a tbd with llvm-readtapi from the generated dylib
# RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4

# (4) Compare the two tbds to make sure they are identical
# RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

#--- foo.s

.global _foo
_foo:
  ret

#--- bar.s

.global _bar
_bar:
  ret

#--- baz.s

.global _baz
_baz:
  ret
