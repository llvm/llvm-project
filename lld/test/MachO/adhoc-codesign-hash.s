# REQUIRES: x86, aarch64
# RUN: rm -rf %t; mkdir -p %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/empty-arm64-macos.o %s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-iossimulator -o %t/empty-arm64-iossimulator.o %s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/empty-x86_64-macos.o %s

# RUN: %lld -arch arm64 -dylib -adhoc_codesign -o %t/empty-arm64-macos.dylib %t/empty-arm64-macos.o
# RUN: %lld -arch arm64 -dylib -adhoc_codesign -o %t/empty-arm64-iossimulator.dylib %t/empty-arm64-iossimulator.o
# RUN: %lld -arch x86_64 -dylib -adhoc_codesign -o %t/empty-x86_64-macos.dylib %t/empty-x86_64-macos.o

# RUN: obj2yaml %t/empty-arm64-macos.dylib | FileCheck %s -D#DATA_OFFSET=16400 -D#DATA_SIZE=304
# RUN: obj2yaml %t/empty-arm64-iossimulator.dylib | FileCheck %s -D#DATA_OFFSET=16400 -D#DATA_SIZE=304
# RUN: obj2yaml %t/empty-x86_64-macos.dylib | FileCheck %s -D#DATA_OFFSET=4112 -D#DATA_SIZE=208

# CHECK:    - cmd:             LC_CODE_SIGNATURE
# CHECK-NEXT: cmdsize:         16
# CHECK-NEXT: dataoff:         [[#DATA_OFFSET]]
# CHECK-NEXT: datasize:        [[#DATA_SIZE]]

# RUN: %python %p/Inputs/code-signature-check.py %t/empty-arm64-macos.dylib 16400 304 0 16400
# RUN: %python %p/Inputs/code-signature-check.py %t/empty-arm64-iossimulator.dylib 16400 304 0 16400
# RUN: %python %p/Inputs/code-signature-check.py %t/empty-x86_64-macos.dylib 4112 208 0 4112
