// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-gnu -verify %s

struct {
    int a;
}
// expected-note@+1 {{conflicting attribute is here}}
__attribute__((gcc_struct))
// expected-error@+1 {{'ms_struct' and 'gcc_struct' attributes are not compatible}}
__attribute__((ms_struct))
t1;

struct {
    int a;
}
// expected-note@+1 {{conflicting attribute is here}}
__attribute__((ms_struct))
// expected-error@+1 {{'gcc_struct' and 'ms_struct' attributes are not compatible}}
__attribute__((gcc_struct))
t2;

#pragma ms_struct on
struct {
    int a;
}
// No diagnostic for an attribute, unambiguously overriding the pragma.
__attribute__((gcc_struct))
t3;

struct {
    int a;
}
// expected-note@+1 {{conflicting attribute is here}}
__attribute__((ms_struct))
// expected-error@+1 {{'gcc_struct' and 'ms_struct' attributes are not compatible}}
__attribute__((gcc_struct))
t4;
#pragma ms_struct off
