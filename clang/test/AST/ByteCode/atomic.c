// RUN: %clang_cc1 -fgnuc-version=4.2.1 -triple=i686-linux-gnu -ffreestanding -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1 -fgnuc-version=4.2.1 -triple=i686-linux-gnu -ffreestanding -verify=both,ref %s

/// FIXME: Copied from test/Sema/atomic-expr.c.
/// this expression seems to be rejected for weird reasons,
/// but we imitate the current interpreter's behavior.
_Atomic int ai = 0;
// FIXME: &ai is an address constant, so this should be accepted as an
// initializer, but the bit-cast inserted due to the pointer conversion is
// tripping up the test for whether the initializer is a constant expression.
// The warning is correct but the error is not.
_Atomic(int *) aip3 = &ai; // both-warning {{incompatible pointer types initializing '_Atomic(int *)' with an expression of type '_Atomic(int) *'}} \
                           // both-error {{initializer element is not a compile-time constant}}

#include <stdatomic.h>



_Static_assert(__GCC_ATOMIC_BOOL_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_BOOL_LOCK_FREE == __CLANG_ATOMIC_BOOL_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_CHAR_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR_LOCK_FREE == __CLANG_ATOMIC_CHAR_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_CHAR16_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR16_T_LOCK_FREE == __CLANG_ATOMIC_CHAR16_T_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_CHAR32_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_CHAR32_T_LOCK_FREE == __CLANG_ATOMIC_CHAR32_T_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_WCHAR_T_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_WCHAR_T_LOCK_FREE == __CLANG_ATOMIC_WCHAR_T_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_SHORT_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_SHORT_LOCK_FREE == __CLANG_ATOMIC_SHORT_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_INT_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_INT_LOCK_FREE == __CLANG_ATOMIC_INT_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_LONG_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_LONG_LOCK_FREE == __CLANG_ATOMIC_LONG_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_LLONG_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_LLONG_LOCK_FREE == __CLANG_ATOMIC_LLONG_LOCK_FREE, "");
_Static_assert(__GCC_ATOMIC_POINTER_LOCK_FREE == 2, "");
_Static_assert(__GCC_ATOMIC_POINTER_LOCK_FREE == __CLANG_ATOMIC_POINTER_LOCK_FREE, "");

_Static_assert(__c11_atomic_is_lock_free(1), "");
_Static_assert(__c11_atomic_is_lock_free(2), "");
_Static_assert(__c11_atomic_is_lock_free(3), ""); // both-error {{not an integral constant expression}}
_Static_assert(__c11_atomic_is_lock_free(4), "");
_Static_assert(__c11_atomic_is_lock_free(8), "");
_Static_assert(__c11_atomic_is_lock_free(16), ""); // both-error {{not an integral constant expression}}
_Static_assert(__c11_atomic_is_lock_free(17), ""); // both-error {{not an integral constant expression}}

_Static_assert(__atomic_is_lock_free(1, 0), "");
_Static_assert(__atomic_is_lock_free(2, 0), "");
_Static_assert(__atomic_is_lock_free(3, 0), ""); // both-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(4, 0), "");
_Static_assert(__atomic_is_lock_free(8, 0), "");
_Static_assert(__atomic_is_lock_free(16, 0), ""); // both-error {{not an integral constant expression}}
_Static_assert(__atomic_is_lock_free(17, 0), ""); // both-error {{not an integral constant expression}}

_Static_assert(atomic_is_lock_free((atomic_char*)0), "");
_Static_assert(atomic_is_lock_free((atomic_short*)0), "");
_Static_assert(atomic_is_lock_free((atomic_int*)0), "");
_Static_assert(atomic_is_lock_free((atomic_long*)0), "");
_Static_assert(atomic_is_lock_free(0 + (atomic_char*)0), "");

_Static_assert(__atomic_always_lock_free(1, (void*)1), "");
_Static_assert(__atomic_always_lock_free(1, (void*)-1), "");
_Static_assert(!__atomic_always_lock_free(4, (void*)2), "");
_Static_assert(!__atomic_always_lock_free(4, (void*)-2), "");
_Static_assert(__atomic_always_lock_free(4, (void*)4), "");
_Static_assert(__atomic_always_lock_free(4, (void*)-4), "");

_Static_assert(__atomic_always_lock_free(1, "string"), "");
_Static_assert(!__atomic_always_lock_free(2, "string"), "");
_Static_assert(__atomic_always_lock_free(2, (int[2]){}), "");
void dummyfn();
_Static_assert(__atomic_always_lock_free(2, dummyfn) || 1, "");
