# REQUIRES: system-windows, x86

# Test that a response.txt file always uses / instead of \.
# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build
# RUN: llvm-mc %s -o %t.dir/build/foo.o -filetype=obj -triple=x86_64-pc-linux
# RUN: cd %t.dir
# RUN: ld.lld build/foo.o --reproduce repro.tar
# RUN: tar xOf repro.tar repro/response.txt | FileCheck -DPATH='%:t.dir' %s

# CHECK: [[PATH]]/build/foo.o
