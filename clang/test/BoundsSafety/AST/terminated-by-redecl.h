#pragma clang system_header

#include <ptrcheck.h>

void test_system_no_annot_argument(int *p);
void test_system_nt_argument(int *__null_terminated p);
void test_system_nt_argument_implicit_1(const char *p);
void test_system_nt_argument_implicit_2(const char *p);
void test_system_nt_argument_implicit_3(const char *p);

int *test_system_no_annot_return();
int *__null_terminated test_system_nt_return();
const char *test_system_nt_return_implicit_1();
const char *test_system_nt_return_implicit_2();
const char *test_system_nt_return_implicit_3();
