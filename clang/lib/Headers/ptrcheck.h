/*===---- ptrcheck.h - Pointer bounds hints & specifications ----------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __PTRCHECK_H
#define __PTRCHECK_H

/* __has_ptrcheck can be used in preprocessor macros (and other parts of the
   language expecting constant expressions) to test if bounds attributes
   exist. */
#if defined(__has_feature) && __has_feature(bounds_attributes)
  #define __has_ptrcheck 1
#else
  #define __has_ptrcheck 0
#endif

#if defined(__has_feature) && __has_feature(bounds_safety_attributes)
  #define __has_bounds_safety_attributes 1
#else
  #define __has_bounds_safety_attributes 0
#endif

#if __has_ptrcheck || __has_bounds_safety_attributes

/* An attribute that modifies a pointer type such than it has the ABI of a
   regular C pointer, without allowing pointer arithmetic. Pointer arithmetic is
   a compile-time error. A __single pointer is expected to be either NULL or
   point to exactly one valid value. */
#define __single __attribute__((__single__))

/* An attribute that modifies a pointer type such than it can be used exactly
   like a regular C pointer, with unchecked arithmetic and dereferencing. An
   __unsafe_indexable pointer cannot convert implicitly to another type of
   pointer since that would require information that is not available to the
   program. You must use __unsafe_forge_bidi_indexable or __unsafe_forge_single
   to convert __unsafe_indexable pointers to so-called safe pointers. */
#define __unsafe_indexable __attribute__((__unsafe_indexable__))

/* An attribute that modifies a pointer type such that it has the ABI of a
   regular C pointer, but it implicitly converts to a __bidi_indexable pointer
   with bounds that assume there are N valid elements starting at its address.
   The conversion happens at the same point the object converts to an rvalue, or
   immediately for values which cannot be lvalues (such as function calls). */

/* Assignments to the pointer object must be accompanied with an assignment to
   N if it is assignable. */

/* N must either be an expression that evaluates to a constant, or an integer
   declaration from the same scope, or (for structure fields) a declaration
   contained in basic arithmetic. */
#define __counted_by(N) __attribute__((__counted_by__(N)))

/* Identical to __counted_by_or_null(N), aside that the pointer may be null for
 * non-zero values of N. */
#define __counted_by_or_null(N) __attribute__((__counted_by_or_null__(N)))

/* Identical to __counted_by(N), aside that N is a byte count instead of an
   object count. */
#define __sized_by(N) __attribute__((__sized_by__(N)))

/* Identical to __sized_by(N), aside that the pointer may be null for non-zero
 * values of N. */
#define __sized_by_or_null(N) __attribute__((__sized_by_or_null__(N)))

/* An attribute that modifies a pointer type such that it has the ABI of a
   regular C pointer, but it implicitly converts to a __bidi_indexable pointer
   with bounds that assume that E is one-past-the-end of the original object.
   Implicitly, referencing E in the same scope will create a pointer that
   converts to a __bidi_indexable pointer one-past-the-end of the original
   object, but with a lower bound set to the value of the pointer that is
   attributed. */

/* Assignments to the pointer object must be accompanied with an assignment to
   E if it is assignable. */
#define __ended_by(E) __attribute__((__ended_by__(E)))

/* The __terminated_by(T) attribute can be applied to arrays and pointers. The
   argument T specifies the terminator and must be an integer constant
   expression. Even though T has to be an integer constant, __terminated_by(T)
   can be applied to pointer arrays as well. For convenience, the
   __null_terminated macro is provided, which is equivalent to
   __terminated_by(0).

   The __terminated_by(T) attribute can be applied only to __single pointers. If
   the pointer attribute is not specified, it is automatically set to __single.
   A __terminated_by(T) pointer points to the first element of an array that is
   terminated with T.

   Arithmetic on __terminated_by(T) pointers is restricted to only incrementing
   the pointer by one, and must be able to be evaluated at compile-time.
   Pointer arithmetic generates a runtime check to ensure that the pointer
   doesn't point pass the terminator.

   A __terminated_by(T) pointer has the ABI of a regular C pointer.

   When __terminated_by(T) is applied to an array, the compiler checks if the
   array is terminated with the given terminator T during the initialization.
   Moreover, a __terminated_by(T) array decays to a __terminated_by(T) __single
   pointer, instead of decaying to a __bidi_indexable pointer. */
#define __terminated_by(T) __attribute__((__terminated_by__(T)))
#define __null_terminated __terminated_by(0)

/* Directives that tells the compiler to assume that subsequent pointer types
   have the ABI specified by the ABI parameter, which may be one of single,
   indexable, bidi_indexable or unsafe_indexable. */

/* In project files, the ABI is assumed to be single by default. In headers
   included from libraries or the SDK, the ABI is assumed to be unsafe_indexable
   by default. */
#define __ptrcheck_abi_assume_single() \
  _Pragma("clang abi_ptr_attr set(single)")

#define __ptrcheck_abi_assume_unsafe_indexable() \
  _Pragma("clang abi_ptr_attr set(unsafe_indexable)")

/* Create a __single pointer of a given type (T), starting at address P. T must
   be a pointer type. */
#define __unsafe_forge_single(T, P) \
  ((T __single)__builtin_unsafe_forge_single((P)))

/* Create a __terminated_by pointer of a given pointer type (T), starting at
   address P, terminated by E. T must be a pointer type. */
#define __unsafe_forge_terminated_by(T, P, E)                                  \
  ((T __terminated_by(E))__builtin_unsafe_forge_terminated_by((P), (E)))

/* Create a __terminated_by pointer of a given pointer type (T), starting at
   address P, terminated by 0. T must be a pointer type. */
#define __unsafe_forge_null_terminated(T, P) __unsafe_forge_terminated_by(T, P, 0)

/* Instruct the compiler to disregard the bounds of an array used in a function
   prototype and allow the decayed pointer to use __counted_by. This is a niche
   capability that is only useful in limited patterns (the way that `mig` uses
   arrays being one of them). */
#define __array_decay_discards_count_in_parameters \
  __attribute__((__decay_discards_count_in_parameters__))

/* An attribute to indicate a variable to be effectively constant (or data const)
   that it is allocated in a const section so cannot be modified after an early
   stage of bootup, for example. Adding this attribute allows a global variable
   to be used in __counted_by attribute of struct fields, function parameter, or
   local variable just like actual constants.
   Note that ensuring the value never changes once it is used is the user's
   responsibility. One way to achieve this is the xnu model, in which certain
   variables are placed in a segment that is remapped as read-only after
   initialization. */
#define __unsafe_late_const __attribute__((__unsafe_late_const__))

#else

#define __single
#define __unsafe_indexable
#define __counted_by(N)
#define __counted_by_or_null(N)
#define __sized_by(N)
#define __sized_by_or_null(N)
#define __ended_by(E)

/* We intentionally define the terminated_by attributes to nothing. */
#define __terminated_by(T)
#define __null_terminated

/* Similarly, we intentionally define to nothing the
   __ptrcheck_abi_assume_single and __ptrcheck_abi_assume_unsafe_indexable
   macros because they do not lead to an ABI incompatibility. However, we do not
   define the indexable and unsafe_indexable ones because the diagnostic is
   better than the silent ABI break. */
#define __ptrcheck_abi_assume_single()
#define __ptrcheck_abi_assume_unsafe_indexable()

/* __unsafe_forge intrinsics are defined as regular C casts. */
#define __unsafe_forge_single(T, P) ((T)(P))
#define __unsafe_forge_terminated_by(T, P, E) ((T)(P))
#define __unsafe_forge_null_terminated(T, P) ((T)(P))

/* decay operates normally; attribute is meaningless without pointer checks. */
#define __array_decay_discards_count_in_parameters

#endif /* __has_ptrcheck || __has_bounds_safety_attributes */

#if __has_ptrcheck

/* An attribute that modifies a pointer type such that its ABI is three pointer
   components: the pointer value itself (the pointer value); one-past-the-end of
   the object it is derived from (the upper bound); and the base address of the
   object it is derived from (the lower bound). The pointer value is allowed to
   lie outside the [lower bound, upper bound) interval, and it supports the
   entire range of arithmetic operations that are usually applicable to
   pointers. Bounds are implicitly checked only when the pointer is dereferenced
   or converted to a different representation. */
#define __bidi_indexable __attribute__((__bidi_indexable__))

/* An attribute that modifies a pointer type such that its ABI is two pointer
   components: the pointer value itself (the lower bound); and one-past-the-end
   of the object it is derived from (the upper bound). Indexable pointers do not
   support negative arithmetic operations: it is a compile-time error to use a
   subtraction or add a negative quantity to them, and it is a runtime error if
   the same happens at runtime while it can't be detected at compile-time. Same
   as __bidi_indexable pointers, __indexable pointers are bounds-checked when
   dereferenced or converted to another representation. */
#define __indexable __attribute__((__indexable__))

/* Directives that tells the compiler to assume that subsequent pointer types
   have the ABI specified by the ABI parameter, which may be one of single,
   indexable, bidi_indexable or unsafe_indexable. */

#define __ptrcheck_abi_assume_indexable() \
  _Pragma("clang abi_ptr_attr set(indexable)")

#define __ptrcheck_abi_assume_bidi_indexable() \
  _Pragma("clang abi_ptr_attr set(bidi_indexable)")

/* Create a __bidi_indexable pointer of a given pointer type (T), starting at
   address P, pointing to S bytes of valid memory. T must be a pointer type. */
#define __unsafe_forge_bidi_indexable(T, P, S) \
  ((T __bidi_indexable)__builtin_unsafe_forge_bidi_indexable((P), (S)))

/* Create a wide pointer with the same lower bound and upper bounds as X, but
   with a pointer component also equal to the lower bound. */
#define __ptr_lower_bound(X) __builtin_get_pointer_lower_bound(X)

/* Create a wide pointer with the same lower bound and upper bounds as X, but
   with a pointer component also equal to the upper bound. */
#define __ptr_upper_bound(X) __builtin_get_pointer_upper_bound(X)

/* Convert a __terminated_by(T) pointer to an __indexable pointer. These
   operations will calculate the upper bound by iterating over the memory
   pointed to by P in order to find the terminator.

   The __terminated_by_to_indexable(P) does NOT include the terminator within
   bounds of the __indexable pointer. Consequently, the terminator cannot be
   erased (or even accessed) through the __indexable pointer. The address one
   past the end of the array (pointing to the terminator) can be found with
   __ptr_upper_bound().

   The __unsafe_terminated_by_to_indexable(P) does include the terminator within
   the bounds of the __indexable pointer. This makes the operation unsafe, since
   the terminator can be erased, and thus using P might result in out-of-bounds
   access. */
#define __terminated_by_to_indexable(P) \
  __builtin_terminated_by_to_indexable(P)
#define __unsafe_terminated_by_to_indexable(P) \
  __builtin_unsafe_terminated_by_to_indexable(P)
#define __null_terminated_to_indexable(P) \
  __builtin_terminated_by_to_indexable((P), 0)
#define __unsafe_null_terminated_to_indexable(P) \
  __builtin_unsafe_terminated_by_to_indexable((P), 0)

/* __unsafe_terminated_by_from_indexable(T, PTR [, PTR_TO_TERM]) converts an
   __indexable pointer to a __terminated_by(T) pointer. The operation will
   check if the given terminator T occurs in the memory pointed to by PTR.
   If so, the operation evaluates to __terminated_by(T) pointer. Otherwise, it
   traps.

   The operation has an optional parameter PTR_TO_TERM, which changes the way
   how the check for the terminator existence is generated. PTR_TO_TERM must
   point to the terminator element and be within the bounds of PTR.
   If PTR_TO_TERM is provided, the runtime will check if it is in fact within
   the bounds and points to an element that equals to T. If PTR_TO_TERM is not
   provided, the runtime will iterate over the memory pointed to by PTR to find
   the terminator.

   The operation is unsafe, since the terminator can be erased through PTR after
   the conversion. This can result in out-of-bounds access through the newly
   created __terminated_by(T) pointer.

   For convenience, the
   __unsafe_null_terminated_from_indexable(PTR [, PTR_TO_TERM]) macro is
   provided, which assumes that the terminator is 0. */
#define __unsafe_terminated_by_from_indexable(T, ...) \
  __builtin_unsafe_terminated_by_from_indexable((T), __VA_ARGS__)
#define __unsafe_null_terminated_from_indexable(...) \
  __builtin_unsafe_terminated_by_from_indexable(0, __VA_ARGS__)

/* An attribute to indicate that a function is unavailable when -fbounds-safety
   is enabled because it is unsafe. Calls to functions annotated with this
   attribute are errors when -fbounds-safety is enabled, but are allowed when
   -fbounds-safety is disabled.

   Example:

   void* __ptrcheck_unavailable some_unsafe_api(void*);
 */
#define __ptrcheck_unavailable                                                 \
  __attribute__((__unavailable__("unavailable with -fbounds-safety.")))

/* __ptrcheck_unavailable_r is the same as __ptrcheck_unavailable but it takes
   as an argument the name of replacement function that is safe for use with
   -fbounds-safety enabled.

   Example:

   void* __counted_by(size) safe_api(void* __counted_by(size), size_t size);

   void* __ptrcheck_unavailable_r(safe_api) some_unsafe_api(void*);
 */
#define __ptrcheck_unavailable_r(REPLACEMENT)                                  \
  __attribute__((__unavailable__(                                                  \
      "unavailable with -fbounds-safety. Use " #REPLACEMENT " instead.")))

#else

/* We intentionally define to nothing pointer attributes which do not have an
   impact on the ABI. __indexable and __bidi_indexable are not defined because
   of the ABI incompatibility that makes the diagnostic preferable. */

/* __unsafe_forge intrinsics are defined as regular C casts. */
#define __unsafe_forge_bidi_indexable(T, P, S) ((T)(P))

/* The conversion between terminated_by pointers just evaluates to the pointer
   argument. */
#define __terminated_by_to_indexable(P) (P)
#define __unsafe_terminated_by_to_indexable(P) (P)
#define __null_terminated_to_indexable(P) (P)
#define __unsafe_null_terminated_to_indexable(P) (P)

#define __IGNORE_REST(P, ...) (P)
/* Adding DUMMY is a workaround for the case where the macro is called
   with only a single argument: calling the __IGNORE_REST macro with only a
   single argument is a C23 extension and emits a warning for earlier
   versions.
 */
#define __unsafe_terminated_by_from_indexable(T, ...)                          \
  __IGNORE_REST(__VA_ARGS__, DUMMY)
#define __unsafe_null_terminated_from_indexable(...)                           \
  __unsafe_terminated_by_from_indexable(DUMMY_TYPE, __VA_ARGS__)

/* The APIs marked with these attributes are available outside the context of
   pointer checks, so do nothing. */
#define __ptrcheck_unavailable
#define __ptrcheck_unavailable_r(REPLACEMENT)

#endif /* __has_ptrcheck */

#endif /* __PTRCHECK_H */
