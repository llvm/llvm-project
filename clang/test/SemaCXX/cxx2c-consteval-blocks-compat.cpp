// RUN: %clang_cc1 -std=c++20 -verify=pre26 -Wc++23-extensions -Wc++20-extensions %s
// RUN: %clang_cc1 -std=c++23 -verify=pre26 -Wc++23-extensions -Wc++20-extensions %s
// RUN: %clang_cc1 -std=c++26 -verify=cxx26 -Wc++23-extensions -Wc++20-extensions -Wpre-c++26-compat %s

// This test also checks that we don't warn about the lambda that we internally
// create for this being 'static' and 'consteval', which are C++23 and C++20
// extensions, even if the corresponding extension warnings are enabled.

consteval {} // pre26-warning {{consteval blocks are a C++2c extension}} \
                cxx26-warning {{consteval blocks are incompatible with C++ standards before C++2c}}
