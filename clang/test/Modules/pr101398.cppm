// RUN: mkdir -p %t
// RUN: %clang -std=c++20 -xc++-module %s -Xclang -verify --precompile -o %t/tmp.pcm
// not modules

// expected-error@* {{missing 'export module' declaration in module interface unit}}
