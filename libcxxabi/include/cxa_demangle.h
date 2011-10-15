//===-------------------------- cxa_demangle.h ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _CXA_DEMANGLE_H
#define _CXA_DEMANGLE_H

#include <cxxabi.h>

namespace __cxxabiv1
{

namespace __libcxxabi
{

class __demangle_tree;
class __node;

char*
__demangle(__demangle_tree, char*, size_t*, int*);

struct __demangle_tree_rv
{
    __demangle_tree* ptr_;

    explicit __demangle_tree_rv(__demangle_tree* ptr)
        : ptr_(ptr) {}
};

class __demangle_tree
{
    const char* __mangled_name_begin_;
    const char* __mangled_name_end_;
    int         __status_;
    __node*     __root_;
    __node*     __node_begin_;
    __node*     __node_end_;
    __node*     __node_cap_;
    __node**    __sub_begin_;
    __node**    __sub_end_;
    __node**    __sub_cap_;
    __node**    __t_begin_;
    __node**    __t_end_;
    __node**    __t_cap_;
    bool        __tag_templates_;
    bool        __fix_forward_references_;
    bool        __owns_buf_;

    __demangle_tree& operator=(const __demangle_tree&);
public:
    __demangle_tree(const char*, char*, size_t);
    ~__demangle_tree();

    __demangle_tree(__demangle_tree&);
    __demangle_tree(__demangle_tree_rv);
    operator __demangle_tree_rv() {return __demangle_tree_rv(this);}

    int __status() const;
    size_t size() const;
    char* __get_demangled_name(char*) const;

    void __parse();

private:
    const char* __parse_encoding(const char*, const char*);
    const char* __parse_type(const char*, const char*,
                             bool = true, bool = false);
    const char* __parse_special_name(const char*, const char*);
    const char* __parse_name(const char*, const char*);
    const char* __parse_bare_function_type(const char*, const char*);
    const char* __parse_call_offset(const char*, const char*);
    const char* __parse_number(const char*, const char*);
    const char* __parse_cv_qualifiers(const char* first, const char* last,
                                      unsigned& cv, bool = false);
    const char* __parse_nested_name(const char*, const char*);
    const char* __parse_discriminator(const char*, const char*);
    const char* __parse_local_name(const char*, const char*);
    const char* __parse_unscoped_template_name(const char*, const char*);
    const char* __parse_unscoped_name(const char*, const char*);
    const char* __parse_operator_name(const char*, const char*, int* = 0);
    const char* __parse_unqualified_name(const char*, const char*);
    const char* __parse_source_name(const char*, const char*);
    const char* __parse_ctor_dtor_name(const char*, const char*);
    const char* __parse_unnamed_type_name(const char*, const char*);
    const char* __parse_template_args(const char*, const char*);
    const char* __parse_template_arg(const char*, const char*);
    const char* __parse_expression(const char*, const char*);
    const char* __parse_expr_primary(const char*, const char*);
    const char* __parse_substitution(const char*, const char*);
    const char* __parse_builtin_type(const char*, const char*);
    const char* __parse_function_type(const char*, const char*);
    const char* __parse_class_enum_type(const char*, const char*);
    const char* __parse_array_type(const char*, const char*);
    const char* __parse_pointer_to_member_type(const char*, const char*);
    const char* __parse_decltype(const char*, const char*);
    const char* __parse_template_param(const char*, const char*);
    const char* __parse_unresolved_name(const char*, const char*);
    const char* __parse_unresolved_type(const char*, const char*);
    const char* __parse_base_unresolved_name(const char*, const char*);
    const char* __parse_simple_id(const char*, const char*);
    const char* __parse_destructor_name(const char*, const char*);
    const char* __parse_function_param(const char*, const char*);
    const char* __parse_const_cast_expr(const char*, const char*);
    const char* __parse_alignof_expr(const char*, const char*);
    const char* __parse_call_expr(const char*, const char*);
    const char* __parse_conversion_expr(const char*, const char*);
    const char* __parse_delete_array_expr(const char*, const char*);
    const char* __parse_delete_expr(const char*, const char*);
    const char* __parse_dynamic_cast_expr(const char*, const char*);
    const char* __parse_dot_star_expr(const char*, const char*);
    const char* __parse_dot_expr(const char*, const char*);
    const char* __parse_decrement_expr(const char*, const char*);
    const char* __parse_new_expr(const char*, const char*);
    const char* __parse_increment_expr(const char*, const char*);
    const char* __parse_arrow_expr(const char*, const char*);
    const char* __parse_reinterpret_cast_expr(const char*, const char*);
    const char* __parse_static_cast_expr(const char*, const char*);
    const char* __parse_sizeof_type_expr(const char*, const char*);
    const char* __parse_sizeof_param_pack_expr(const char*, const char*);
    const char* __parse_typeid_expr(const char*, const char*);
    const char* __parse_throw_expr(const char*, const char*);
    const char* __parse_pack_expansion(const char*, const char*);
    const char* __parse_sizeof_function_param_pack_expr(const char*, const char*);
    const char* __parse_dot_suffix(const char*, const char*);
    const char* __parse_unresolved_qualifier_level(const char*, const char*);
    const char* __parse_hex_number(const char*, const char*, unsigned long long&);

    template <class _Tp> bool __make();
    template <class _Tp, class _A0> bool __make(_A0 __a0);
    template <class _Tp, class _A0, class _A1> bool __make(_A0 __a0, _A1 __a1);
    template <class _Tp, class _A0, class _A1, class _A2>
        bool __make(_A0 __a0, _A1 __a1, _A2 __a2);
    template <class _Tp, class _A0, class _A1, class _A2, class _A3>
        bool __make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3);
    template <class _Tp, class _A0, class _A1, class _A2, class _A3, class _A4>
        bool __make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3, _A4 __a4);
    template <class _Tp, class _A0, class _A1, class _A2, class _A3, class _A4,
                         class _A5>
        bool __make(_A0 __a0, _A1 __a1, _A2 __a2, _A3 __a3, _A4 __a4, _A5 __a5);

    friend
    char*
    __demangle(__demangle_tree, char*, size_t*, int*);

};

__demangle_tree
__demangle(const char*);

__demangle_tree
__demangle(const char*, char*, size_t);

}  // __libcxxabi
}  // __cxxabiv1

#endif  // _CXA_DEMANGLE_H
