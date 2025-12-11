// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-gnu -verify %s

struct {
    int a;
}
// expected-note@+1 {{conflicting attribute is here}}
__attribute__((gcc_struct))
// expected-error@+1 {{'ms_struct' and 'gcc_struct' attributes are not compatible}}
__attribute__((ms_struct))
t1;
