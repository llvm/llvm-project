# REQUIRES: x86, system-linux

# Test that we don't erroneously replace \ with / on UNIX, as it's
# legal for a filename to contain backslashes.
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: llvm-mc %s -o %t.dir/foo\\.o -filetype=obj -triple=x86_64-pc-linux
# RUN: ld.lld %t.dir/foo\\.o --reproduce %t.dir/repro.tar -o /dev/null
# RUN: tar tf %t.dir/repro.tar | FileCheck %s

# CHECK: repro/{{.*}}/foo\{{[\]?}}.o
