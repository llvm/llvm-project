//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Check that user-specializations are diagnosed
// See [meta.rqmts]/4, [meta.trans.other]/5, [meta.trans.other]/7

#include <type_traits>

#include "test_macros.h"

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else
struct S {};

#  define SPECIALIZE_TRAIT(Trait)                                                                                      \
    template <>                                                                                                        \
    struct std::Trait<S>

SPECIALIZE_TRAIT(add_const);            // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(add_cv);               // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(add_volatile);         // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(add_lvalue_reference); // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(add_rvalue_reference); // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(add_pointer);          // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(decay);                // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(invoke_result);        // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(make_unsigned);        // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_all_extents);   // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_const);         // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_cv);            // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_extent);        // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_pointer);       // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_reference);     // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(remove_volatile);      // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(underlying_type);      // expected-error {{cannot be specialized}}

#  if TEST_STD_VER <= 17
SPECIALIZE_TRAIT(result_of); // expected-error {{cannot be specialized}}
#  endif

#  if TEST_STD_VER >= 20
SPECIALIZE_TRAIT(remove_cvref);     // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(type_identity);    // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(unwrap_reference); // expected-error {{cannot be specialized}}
SPECIALIZE_TRAIT(unwrap_ref_decay); // expected-error {{cannot be specialized}}
#  endif

#  undef SPECIALIZE_TRAIT
#  define SPECIALIZE_UTT(Trait)                                                                                        \
    template <>                                                                                                        \
    struct std::Trait<S>;                                                                                              \
    template <>                                                                                                        \
    inline constexpr bool std::Trait##_v<S> = false

#  define SPECIALIZE_BTT(Trait)                                                                                        \
    template <>                                                                                                        \
    struct std::Trait<S, S>;                                                                                           \
    template <>                                                                                                        \
    inline constexpr bool std::Trait##_v<S, S> = false

SPECIALIZE_UTT(alignment_of);                       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(conjunction);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(disjunction);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(extent);                             // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(has_unique_object_representations);  // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_abstract);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_aggregate);                       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_arithmetic);                      // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_array);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_assignable);                      // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_base_of);                         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_class);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_compound);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_const);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_constructible);                   // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_convertible);                     // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_copy_assignable);                 // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_copy_constructible);              // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_default_constructible);           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_destructible);                    // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_empty);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_enum);                            // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_final);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_floating_point);                  // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_function);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_fundamental);                     // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_integral);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_invocable);                       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_invocable_r);                     // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_lvalue_reference);                // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_member_pointer);                  // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_member_object_pointer);           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_member_function_pointer);         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_move_assignable);                 // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_move_constructible);              // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_nothrow_assignable);              // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_constructible);           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_copy_assignable);         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_copy_constructible);      // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_default_constructible);   // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_destructible);            // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_move_assignable);         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_move_constructible);      // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_invocable);               // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_nothrow_invocable_r);             // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_nothrow_swappable);               // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_nothrow_swappable_with);          // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_null_pointer);                    // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_object);                          // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_pod);                             // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_pointer);                         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_polymorphic);                     // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_reference);                       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_rvalue_reference);                // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_same);                            // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_scalar);                          // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_signed);                          // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_standard_layout);                 // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_swappable);                       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_swappable_with);                  // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivial);                         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_trivially_assignable);            // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_constructible);         // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_copy_assignable);       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_copy_constructible);    // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_copyable);              // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_default_constructible); // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_destructible);          // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_move_assignable);       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_trivially_move_constructible);    // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_union);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_unsigned);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_void);                            // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_volatile);                        // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(negation);                           // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(rank);                               // expected-error 2 {{cannot be specialized}}

#  if TEST_STD_VER <= 17
SPECIALIZE_UTT(is_literal_type); // expected-error 2 {{cannot be specialized}}
#  endif

#  if TEST_STD_VER >= 20
SPECIALIZE_UTT(is_bounded_array);       // expected-error 2 {{cannot be specialized}}
SPECIALIZE_BTT(is_nothrow_convertible); // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_unbounded_array);     // expected-error 2 {{cannot be specialized}}
#  endif

#  if TEST_STD_VER >= 23
SPECIALIZE_UTT(is_implicit_lifetime); // expected-error 2 {{cannot be specialized}}
SPECIALIZE_UTT(is_scoped_enum);       // expected-error 2 {{cannot be specialized}}
#  endif

#  if TEST_STD_VER >= 26
SPECIALIZE_BTT(is_virtual_base_of); // expected-error 2 {{cannot be specialized}}
#  endif

#  undef SPECIALIZE_UTT
#  undef SPECIALIZE_BTT

template <>
struct std::aligned_storage<1, 3>; // expected-error {{cannot be specialized}}

template <>
struct std::aligned_union<1, S>; // expected-error {{cannot be specialized}}

template <>
struct std::conditional<true, S, S>; // expected-error {{cannot be specialized}}

template <>
struct std::enable_if<true, S>; // expected-error {{cannot be specialized}}

#if TEST_STD_VER >= 20
template <>
struct std::integral_constant<S, {}>; // expected-error {{cannot be specialized}}
#endif
#endif
