// RUN: rm -rf %t
// RUN: %clang_cc1 -fno-objc-infer-related-result-type -Wmodule-build -fmodule-cache-path %t -F %S/Inputs -verify %s

__import_module__ Module; // expected-warning{{building module 'Module' from source}}

