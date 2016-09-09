# REQUIRES: x86, cpio

# Test that a repro archive always uses / instead of \.
# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build
# RUN: llvm-mc %s -o %t.dir/build/foo.o -filetype=obj -triple=x86_64-pc-linux
# RUN: cd %t.dir
# RUN: ld.lld build/foo.o --reproduce repro
# RUN: cpio -t < repro.cpio | FileCheck %s

# CHECK: repro/response.txt
# CHECK: repro/{{.*}}/build/foo.o
