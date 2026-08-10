// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -emit-pch -o %t1.pch -fmodules-cache-path=%t.mcp -I %S/Inputs/redecls
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -emit-pch -o %t2.pch -include-pch %t1.pch -fmodules-cache-path=%t.mcp -I %S/Inputs/redecls
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only -include-pch %t2.pch -I %S/Inputs/redecls -fmodules-cache-path=%t.mcp -verify

#ifndef HEADER1
#define HEADER1

@import a;

#elif !defined(HEADER2)
#define HEADER2

@class AA;
@import b;

#else

@interface SS : AA
@end

#warning parsed this
#endif
// expected-warning@-2{{parsed this}}
