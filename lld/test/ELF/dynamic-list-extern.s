# REQUIRES: x86

# Test that we can parse multiple externs.

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: echo '{ extern "C" { foo; }; extern "C++" { bar; }; };' > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o -shared -o %t.so

# RUN: echo '{ extern "C" { foo }; extern "C++" { bar }; };' > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o -shared -o %t.so

# RUN: echo '{ extern "C++" { std::foo; }; };' > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o -shared -o %t.so

# RUN: echo '{ extern "C++" { std:foo; }; };' > a.list
# RUN: not ld.lld --dynamic-list a.list %t.o -shared 2>&1 | FileCheck %s --check-prefix=ERR-COLON
# RUN: echo '{ extern "C++" { std:::foo; }; };' > a.list
# RUN: not ld.lld --dynamic-list a.list %t.o -shared 2>&1 | FileCheck %s --check-prefix=ERR-COLON
# ERR-COLON: error: a.list:1: ; expected, but got :
