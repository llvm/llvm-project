// RUN: %clang_cc1 -fsyntax-only -fobjc-weak -fobjc-runtime-has-weak -verify -std=c++11 %s -Wno-deprecated-builtins
// expected-no-diagnostics

// Check the results of the various type-trait query functions on
// lifetime-qualified types in ObjC Weak.

#define TRAIT_IS_TRUE(Trait, Type) static_assert(Trait(Type), "")
#define TRAIT_IS_FALSE(Trait, Type) static_assert(!Trait(Type), "")
#define TRAIT_IS_TRUE_2(Trait, Type1, Type2) static_assert(Trait(Type1, Type2), "")
#define TRAIT_IS_FALSE_2(Trait, Type1, Type2) static_assert(!Trait(Type1, Type2), "")

struct HasStrong { id obj; };
struct HasWeak { __weak id obj; };
struct HasUnsafeUnretained { __unsafe_unretained id obj; };

// __has_nothrow_assign
TRAIT_IS_TRUE(__has_nothrow_assign, __strong id);
TRAIT_IS_TRUE(__has_nothrow_assign, __weak id);
TRAIT_IS_TRUE(__has_nothrow_assign, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_assign, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_nothrow_assign, HasStrong);
TRAIT_IS_TRUE(__has_nothrow_assign, HasWeak);
TRAIT_IS_TRUE(__has_nothrow_assign, HasUnsafeUnretained);

// __has_nothrow_copy
TRAIT_IS_TRUE(__has_nothrow_copy, __strong id);
TRAIT_IS_TRUE(__has_nothrow_copy, __weak id);
TRAIT_IS_TRUE(__has_nothrow_copy, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_copy, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_nothrow_copy, HasStrong);
TRAIT_IS_TRUE(__has_nothrow_copy, HasWeak);
TRAIT_IS_TRUE(__has_nothrow_copy, HasUnsafeUnretained);

// __has_nothrow_constructor
TRAIT_IS_TRUE(__has_nothrow_constructor, __strong id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __weak id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_nothrow_constructor, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_nothrow_constructor, HasStrong);
TRAIT_IS_TRUE(__has_nothrow_constructor, HasWeak);
TRAIT_IS_TRUE(__has_nothrow_constructor, HasUnsafeUnretained);

// __has_trivial_assign
TRAIT_IS_TRUE(__has_trivial_assign, __strong id);
TRAIT_IS_FALSE(__has_trivial_assign, __weak id);
TRAIT_IS_TRUE(__has_trivial_assign, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_assign, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_trivial_assign, HasStrong);
TRAIT_IS_FALSE(__has_trivial_assign, HasWeak);
TRAIT_IS_TRUE(__has_trivial_assign, HasUnsafeUnretained);

// __has_trivial_copy
TRAIT_IS_TRUE(__has_trivial_copy, __strong id);
TRAIT_IS_FALSE(__has_trivial_copy, __weak id);
TRAIT_IS_TRUE(__has_trivial_copy, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_copy, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_trivial_copy, HasStrong);
TRAIT_IS_FALSE(__has_trivial_copy, HasWeak);
TRAIT_IS_TRUE(__has_trivial_copy, HasUnsafeUnretained);

// __has_trivial_constructor
TRAIT_IS_TRUE(__has_trivial_constructor, __strong id);
TRAIT_IS_FALSE(__has_trivial_constructor, __weak id);
TRAIT_IS_TRUE(__has_trivial_constructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_constructor, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_trivial_constructor, HasStrong);
TRAIT_IS_FALSE(__has_trivial_constructor, HasWeak);
TRAIT_IS_TRUE(__has_trivial_constructor, HasUnsafeUnretained);

// __has_trivial_destructor
TRAIT_IS_TRUE(__has_trivial_destructor, __strong id);
TRAIT_IS_FALSE(__has_trivial_destructor, __weak id);
TRAIT_IS_TRUE(__has_trivial_destructor, __autoreleasing id);
TRAIT_IS_TRUE(__has_trivial_destructor, __unsafe_unretained id);
TRAIT_IS_TRUE(__has_trivial_destructor, HasStrong);
TRAIT_IS_FALSE(__has_trivial_destructor, HasWeak);
TRAIT_IS_TRUE(__has_trivial_destructor, HasUnsafeUnretained);

// __is_literal
TRAIT_IS_TRUE(__is_literal, __strong id);
TRAIT_IS_TRUE(__is_literal, __weak id);
TRAIT_IS_TRUE(__is_literal, __autoreleasing id);
TRAIT_IS_TRUE(__is_literal, __unsafe_unretained id);

// __is_literal_type
TRAIT_IS_TRUE(__is_literal_type, __strong id);
TRAIT_IS_TRUE(__is_literal_type, __weak id);
TRAIT_IS_TRUE(__is_literal_type, __autoreleasing id);
TRAIT_IS_TRUE(__is_literal_type, __unsafe_unretained id);

// __is_pod
TRAIT_IS_TRUE(__is_pod, __strong id);
TRAIT_IS_FALSE(__is_pod, __weak id);
TRAIT_IS_TRUE(__is_pod, __autoreleasing id);
TRAIT_IS_TRUE(__is_pod, __unsafe_unretained id);
TRAIT_IS_TRUE(__is_pod, HasStrong);
TRAIT_IS_FALSE(__is_pod, HasWeak);
TRAIT_IS_TRUE(__is_pod, HasUnsafeUnretained);

// __is_trivial
TRAIT_IS_TRUE(__is_trivial, __strong id);
TRAIT_IS_FALSE(__is_trivial, __weak id);
TRAIT_IS_TRUE(__is_trivial, __autoreleasing id);
TRAIT_IS_TRUE(__is_trivial, __unsafe_unretained id);
TRAIT_IS_TRUE(__is_trivial, HasStrong);
TRAIT_IS_FALSE(__is_trivial, HasWeak);
TRAIT_IS_TRUE(__is_trivial, HasUnsafeUnretained);

// __is_scalar
TRAIT_IS_TRUE(__is_scalar, __strong id);
TRAIT_IS_FALSE(__is_scalar, __weak id);
TRAIT_IS_TRUE(__is_scalar, __autoreleasing id);
TRAIT_IS_TRUE(__is_scalar, __unsafe_unretained id);

// __is_standard_layout
TRAIT_IS_TRUE(__is_standard_layout, __strong id);
TRAIT_IS_TRUE(__is_standard_layout, __weak id);
TRAIT_IS_TRUE(__is_standard_layout, __autoreleasing id);
TRAIT_IS_TRUE(__is_standard_layout, __unsafe_unretained id);

// __is_trivally_assignable
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __strong id&, __unsafe_unretained id&&);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __strong id);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __weak id);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __autoreleasing id);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __unsafe_unretained id);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __strong id&&);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __weak id&&);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __autoreleasing id&&);
TRAIT_IS_FALSE_2(__is_trivially_assignable, __weak id&, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __autoreleasing id&, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, __unsafe_unretained id&, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_assignable, HasStrong&, HasStrong);
TRAIT_IS_TRUE_2(__is_trivially_assignable, HasStrong&, HasStrong&&);
TRAIT_IS_FALSE_2(__is_trivially_assignable, HasWeak&, HasWeak);
TRAIT_IS_FALSE_2(__is_trivially_assignable, HasWeak&, HasWeak&&);
TRAIT_IS_TRUE_2(__is_trivially_assignable, HasUnsafeUnretained&, HasUnsafeUnretained);
TRAIT_IS_TRUE_2(__is_trivially_assignable, HasUnsafeUnretained&, HasUnsafeUnretained&&);

// __is_trivally_constructible
TRAIT_IS_TRUE(__is_trivially_constructible, __strong id);
TRAIT_IS_FALSE(__is_trivially_constructible, __weak id);
TRAIT_IS_TRUE(__is_trivially_constructible, __autoreleasing id);
TRAIT_IS_TRUE(__is_trivially_constructible, __unsafe_unretained id);

TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __strong id, __unsafe_unretained id&&);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __strong id);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __weak id);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __autoreleasing id);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __unsafe_unretained id);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __strong id&&);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __weak id&&);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __autoreleasing id&&);
TRAIT_IS_FALSE_2(__is_trivially_constructible, __weak id, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __autoreleasing id, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __strong id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __weak id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __autoreleasing id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __unsafe_unretained id);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __strong id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __weak id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __autoreleasing id&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, __unsafe_unretained id, __unsafe_unretained id&&);

TRAIT_IS_TRUE_2(__is_trivially_constructible, HasStrong, HasStrong);
TRAIT_IS_TRUE_2(__is_trivially_constructible, HasStrong, HasStrong&&);
TRAIT_IS_FALSE_2(__is_trivially_constructible, HasWeak, HasWeak);
TRAIT_IS_FALSE_2(__is_trivially_constructible, HasWeak, HasWeak&&);
TRAIT_IS_TRUE_2(__is_trivially_constructible, HasUnsafeUnretained, HasUnsafeUnretained);
TRAIT_IS_TRUE_2(__is_trivially_constructible, HasUnsafeUnretained, HasUnsafeUnretained&&);

// __is_trivially_relocatable
TRAIT_IS_TRUE(__is_trivially_relocatable, __strong id);
TRAIT_IS_FALSE(__is_trivially_relocatable, __weak id);
TRAIT_IS_TRUE(__is_trivially_relocatable, __autoreleasing id);
TRAIT_IS_TRUE(__is_trivially_relocatable, __unsafe_unretained id);
TRAIT_IS_TRUE(__is_trivially_relocatable, HasStrong);
TRAIT_IS_FALSE(__is_trivially_relocatable, HasWeak);
TRAIT_IS_TRUE(__is_trivially_relocatable, HasUnsafeUnretained);
