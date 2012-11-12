// RUN: %clang_cc1 -E %s -o - | FileCheck %s

#if __has_feature(has_nothrow_assign)
int has_nothrow_assign();
#endif
// CHECK: int has_nothrow_assign();

#if __has_feature(has_nothrow_copy)
int has_nothrow_copy();
#endif
// CHECK: int has_nothrow_copy();

#if __has_feature(has_nothrow_constructor)
int has_nothrow_constructor();
#endif
// CHECK: int has_nothrow_constructor();

#if __has_feature(has_trivial_assign)
int has_trivial_assign();
#endif
// CHECK: int has_trivial_assign();

#if __has_feature(has_trivial_copy)
int has_trivial_copy();
#endif
// CHECK: int has_trivial_copy();

#if __has_feature(has_trivial_constructor)
int has_trivial_constructor();
#endif
// CHECK: int has_trivial_constructor();

#if __has_feature(has_trivial_destructor)
int has_trivial_destructor();
#endif
// CHECK: int has_trivial_destructor();

#if __has_feature(has_virtual_destructor)
int has_virtual_destructor();
#endif
// CHECK: int has_virtual_destructor();

#if __has_feature(is_abstract)
int is_abstract();
#endif
// CHECK: int is_abstract();

#if __has_feature(is_base_of)
int is_base_of();
#endif
// CHECK: int is_base_of();

#if __has_feature(is_class)
int is_class();
#endif
// CHECK: int is_class();

#if __has_feature(is_convertible_to)
int is_convertible_to();
#endif
// CHECK: int is_convertible_to();

#if __has_feature(is_empty)
int is_empty();
#endif
// CHECK: int is_empty();

#if __has_feature(is_enum)
int is_enum();
#endif
// CHECK: int is_enum();

#if __has_feature(is_final)
int is_final();
#endif
// CHECK: int is_final();

#if __has_feature(is_pod)
int is_pod();
#endif
// CHECK: int is_pod();

#if __has_feature(is_polymorphic)
int is_polymorphic();
#endif
// CHECK: int is_polymorphic();

#if __has_feature(is_union)
int is_union();
#endif
// CHECK: int is_union();

#if __has_feature(is_literal)
int is_literal();
#endif
// CHECK: int is_literal();

#if __has_feature(is_standard_layout)
int is_standard_layout();
#endif
// CHECK: int is_standard_layout();

#if __has_feature(is_trivially_copyable)
int is_trivially_copyable();
#endif
// CHECK: int is_trivially_copyable();

#if __has_feature(underlying_type)
int underlying_type();
#endif
// CHECK: int underlying_type();
