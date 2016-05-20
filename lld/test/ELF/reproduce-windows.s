# REQUIRES: x86, system-windows

# Test that we can create a repro archive on windows.
# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build
# RUN: llvm-mc %s -o %t.dir/build/foo.o -filetype=obj -triple=x86_64-pc-linux
# RUN: cd %t.dir
# RUN: not ld.lld build/foo.o --reproduce repro
# RUN: cpio -t < repro.cpio | grep -F 'repro\response.txt'
# RUN: cpio -t < repro.cpio | grep -F 'repro\%:t.dir\build\foo.o'
