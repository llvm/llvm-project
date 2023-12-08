// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s

// Function template specializations differing in array type parameter should have unique USRs.

template<class buffer> void foo(buffer);
// CHECK: {{[0-9]+}}:17 | function(Gen,TS)/C++ | foo | c:@F@foo<#{n16C>#*C#
template<> void foo<char[16]>(char[16]);
// CHECK: {{[0-9]+}}:17 | function(Gen,TS)/C++ | foo | c:@F@foo<#{n32C>#*C#
template<> void foo<char[32]>(char[32]);
// CHECK: {{[0-9]+}}:17 | function(Gen,TS)/C++ | foo | c:@F@foo<#{n64C>#*C#
template<> void foo<char[64]>(char[64]);
