#pragma clang system_header

#include <ptrcheck.h>

// expected-note@+1{{previous declaration is here}}
void test_system_nt_argument(int *__null_terminated p);

// expected-note@+1{{previous declaration is here}}
int *__null_terminated test_system_nt_return();

void test_system_nt_argument_impl(int *__null_terminated p);
void test_system_nt_argument_impl(int *p);
