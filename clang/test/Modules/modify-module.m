// Test that if we modify one of the input files used to form a
// header, that module and dependent modules get rebuilt.

// RUN: rm -rf %t
// RUN: mkdir -p %t/include
// RUN: cp %S/Inputs/Modified/A.h %t/include
// RUN: cp %S/Inputs/Modified/B.h %t/include
// RUN: cp %S/Inputs/Modified/module.map %t/include
// RUN: %clang_cc1 -fmodule-cache-path %t/cache -fmodules -I %t/include %s -verify
// expected-no-diagnostics
// RUN: touch %t/include/B.h
// RUN: %clang_cc1 -fmodule-cache-path %t/cache -fmodules -I %t/include %s -verify
// RUN: echo 'int getA(); int getA2();' > %t/include/A.h
// RUN: %clang_cc1 -fmodule-cache-path %t/cache -fmodules -I %t/include %s -verify

@__experimental_modules_import B;

int getValue() { return getA() + getB(); }





