// RUN: rm -rf %t-mcp
// RUN: %clang_cc1 -objcmt-migrate-subscripting -emit-pch -o %t.pch %s -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 -F %S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-mcp -w
// RUN: %clang_cc1 -objcmt-migrate-subscripting -include-pch %t.pch %s -migrate -o %t.remap -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 -F %S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-mcp
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER

@import Module;

#else

#endif
