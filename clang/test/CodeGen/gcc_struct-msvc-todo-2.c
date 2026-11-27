// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-msvc -fms-layout-compatibility=itanium -verify %s

// expected-error@+1 {{Itanium-compatible layout for the Microsoft C++ ABI is not yet supported}}
struct {
    int a;
} t1;
