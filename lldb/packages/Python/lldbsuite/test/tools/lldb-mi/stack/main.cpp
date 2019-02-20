//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <exception>

struct inner
{
    int var_d;
};

struct my_type
{
    int var_a;
    char var_b;
    struct inner inner_;
};

int
local_int_test(void)
{
    int a = 10, b = 20;
    return 0; // BP_local_int_test
}

int
local_int_test_with_args(int c, int d)
{
    int a = 10, b = 20;
    return 0; // BP_local_int_test_with_args
}

int
local_struct_test(void)
{
    struct my_type var_c;
    var_c.var_a = 10;
    var_c.var_b = 'a';
    var_c.inner_.var_d = 30;
    return 0; // BP_local_struct_test
}

int local_struct_test_with_args(struct my_type var_e)
{
    struct my_type var_c;
    var_c.var_a = 10;
    var_c.var_b = 'a';
    var_c.inner_.var_d = 30;
    return 0; // BP_local_struct_test_with_args
}

int
local_array_test(void)
{
    int array[3];
    array[0] = 100;
    array[1] = 200;
    array[2] = 300;
    return 0; // BP_local_array_test
}

int
local_array_test_with_args(int* other_array)
{
    int array[3];
    array[0] = 100;
    array[1] = 200;
    array[2] = 300;
    return 0; // BP_local_array_test_with_args
}

int
local_pointer_test(void)
{
    const char *test_str = "Rakaposhi";
    int var_e = 24;
    int *ptr = &var_e;
    return 0; // BP_local_pointer_test
}

int
local_pointer_test_with_args(const char *arg_str, int *arg_ptr)
{
    const char *test_str = "Rakaposhi";
    int var_e = 24;
    int *ptr = &var_e;
    return 0; // BP_local_pointer_test_with_args
}

int do_tests_with_args()
{
    local_int_test_with_args(30, 40);

    struct my_type var_e;
    var_e.var_a = 20;
    var_e.var_b = 'b';
    var_e.inner_.var_d = 40;
    local_struct_test_with_args(var_e);

    int array[3];
    array[0] = 400;
    array[1] = 500;
    array[2] = 600;
    local_array_test_with_args(array);

    const char *test_str = "String";
    int var_z = 25;
    int *ptr = &var_z;
    local_pointer_test_with_args(test_str, ptr);

    return 0;
}

void catch_unnamed_test()
{
    try
    {
        int i = 1, j = 2;
        throw std::exception(); // BP_catch_unnamed
    }
    catch(std::exception&)
    {
    }
}

int
main(int argc, char const *argv[])
{
    local_int_test();
    local_struct_test();
    local_array_test();
    local_pointer_test();
    catch_unnamed_test();

    do_tests_with_args();
    return 0;
}
