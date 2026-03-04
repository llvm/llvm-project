// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-msvc -verify %s

// expected-error@+1 {{Itanium-compatible layout for the Microsoft C++ ABI is not yet supported}}
struct {
    int a;
} __attribute__((gcc_struct)) t1;
