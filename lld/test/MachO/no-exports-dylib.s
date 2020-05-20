# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -dylib %t.o -o %t.dylib

# RUN: obj2yaml %t.dylib | FileCheck %s
# CHECK: export_size: 0
