//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <string>

struct complex_type
{
    int i;
    struct { long l; } inner;
    complex_type *complex_ptr;
};

struct pcomplex_type : complex_type
{
    pcomplex_type(const complex_type &complx_base, const complex_type &complx_member)
        : complex_type(complx_base), complx(complx_member) { }
    complex_type complx;
    static int si;
};

int pcomplex_type::si;

struct struct_with_unions
{
    struct_with_unions(): u_i(1), u1(-1) {}
    union 
    {
        int u_i;
        int u_j;  
    };
    union 
    {
        int  u1;
        struct
        {
            short s1;
            short s2;
        };
    };
};

void
var_update_test(void)
{
    long l = 1;
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };
    // BP_var_update_test_init

    l = 0;
    // BP_var_update_test_l

    complx.inner.l = 2;
    // BP_var_update_test_complx

    complx_array[1].inner.l = 4;
    // BP_var_update_test_complx_array
}

void
var_list_children_test(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };
    pcomplex_type pcomplx({ 6, { 6L }, &pcomplx}, { 7, { 7L }, &pcomplx});

    // BP_var_list_children_test
}

void
gdb_set_show_print_char_array_as_string_test(void)
{
    const char *cp = "\t\"hello\"\n";
    const char ca[] = "\t\"hello\"\n";
    const char16_t *u16p = u"\t\"hello\"\n";
    const char16_t u16a[] = u"\t\"hello\"\n";
    const char32_t *u32p = U"\t\"hello\"\n";
    const char32_t u32a[] = U"\t\"hello\"\n";

    const char16_t* u16p_rus = u"\\Аламо-сквер";
    const char16_t  u16a_rus[] = u"\\Бейвью";
    const char32_t* u32p_rus = U"\\Чайнатаун";
    const char32_t  u32a_rus[] = U"\\Догпатч";

    // BP_gdb_set_show_print_char_array_as_string_test
}

void
cpp_stl_types_test(void)
{
    std::string std_string = "hello";
    // BP_cpp_stl_types_test
}

void
unnamed_objects_test(void)
{
    struct_with_unions swu;
    // BP_unnamed_objects_test
}

struct not_str
{
    not_str(char _c, int _f)
        : c(_c), f(_f) { }
    char c;
    int f;
};

void
gdb_set_show_print_expand_aggregates(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };
    not_str nstr('a', 0);

    // BP_gdb_set_show_print_expand_aggregates
}

void
gdb_set_show_print_aggregate_field_names(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };

    // BP_gdb_set_show_print_aggregate_field_names
}

int g_MyVar = 3;
static int s_MyVar = 4;

int
main(int argc, char const *argv[])
{
    int a = 10, b = 20;
    s_MyVar = a + b;
    var_update_test();
    var_list_children_test();
    gdb_set_show_print_char_array_as_string_test();
    cpp_stl_types_test();
    unnamed_objects_test();
    gdb_set_show_print_expand_aggregates();
    gdb_set_show_print_aggregate_field_names();
    return 0; // BP_return
}
