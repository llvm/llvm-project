// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -Wauto-import -fmodule-cache-path %t -fmodules -F %S/Inputs %s -verify

#include <NotAModule/NotAModule.h>

@__experimental_modules_import NotAModule; // expected-error{{module 'NotAModule' not found}}


