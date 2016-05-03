# REQUIRES: x86

# Test that we can create a repro archive on windows.
# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: ld.lld --reproduce %t.repro %t.o -o t -shared
# RUN: cpio -t < %t.repro.cpio | FileCheck %s
# CHECK:      {{^[^/\\]*}}.repro{{/|\\}}response.txt
# CHECK-NEXT: .repro{{/|\\}}{{.*}}.o
