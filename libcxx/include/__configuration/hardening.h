//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_HARDENING_H
#define _LIBCPP___CONFIGURATION_HARDENING_H

#include <__config_site>
#include <__configuration/experimental.h>
#include <__configuration/language.h>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

// TODO(LLVM 23): Remove this. We're making these an error to catch folks who might not have migrated.
//       Since hardening went through several changes (many of which impacted user-facing macros),
//       we're keeping these checks around for a bit longer than usual. Failure to properly configure
//       hardening results in checks being dropped silently, which is a pretty big deal.
#if defined(_LIBCPP_ENABLE_ASSERTIONS)
#  error "_LIBCPP_ENABLE_ASSERTIONS has been removed, please use _LIBCPP_HARDENING_MODE=<mode> instead (see docs)"
#endif
#if defined(_LIBCPP_ENABLE_HARDENED_MODE)
#  error "_LIBCPP_ENABLE_HARDENED_MODE has been removed, please use _LIBCPP_HARDENING_MODE=<mode> instead (see docs)"
#endif
#if defined(_LIBCPP_ENABLE_SAFE_MODE)
#  error "_LIBCPP_ENABLE_SAFE_MODE has been removed, please use _LIBCPP_HARDENING_MODE=<mode> instead (see docs)"
#endif
#if defined(_LIBCPP_ENABLE_DEBUG_MODE)
#  error "_LIBCPP_ENABLE_DEBUG_MODE has been removed, please use _LIBCPP_HARDENING_MODE=<mode> instead (see docs)"
#endif

// The library provides the macro `_LIBCPP_HARDENING_MODE` which can be set to one of the following values:
//
// - `_LIBCPP_HARDENING_MODE_NONE`;
// - `_LIBCPP_HARDENING_MODE_FAST`;
// - `_LIBCPP_HARDENING_MODE_EXTENSIVE`;
// - `_LIBCPP_HARDENING_MODE_DEBUG`.
//
// These values have the following effects:
//
// - `_LIBCPP_HARDENING_MODE_NONE` -- sets the hardening mode to "none" which disables all runtime hardening checks;
//
// - `_LIBCPP_HARDENING_MODE_FAST` -- sets that hardening mode to "fast". The fast mode enables security-critical checks
//   that can be done with relatively little runtime overhead in constant time;
//
// - `_LIBCPP_HARDENING_MODE_EXTENSIVE` -- sets the hardening mode to "extensive". The extensive mode is a superset of
//   the fast mode that additionally enables checks that are relatively cheap and prevent common types of logic errors
//   but are not necessarily security-critical;
//
// - `_LIBCPP_HARDENING_MODE_DEBUG` -- sets the hardening mode to "debug". The debug mode is a superset of the extensive
//   mode and enables all checks available in the library, including internal assertions. Checks that are part of the
//   debug mode can be very expensive and thus the debug mode is intended to be used for testing, not in production.

// Inside the library, assertions are categorized so they can be cherry-picked based on the chosen hardening mode. These
// macros are only for internal use -- users should only pick one of the high-level hardening modes described above.
//
// - `_LIBCPP_ASSERT_VALID_INPUT_RANGE` -- checks that ranges (whether expressed as an iterator pair, an iterator and
//   a sentinel, an iterator and a count, or a `std::range`) given as input to library functions are valid:
//   - the sentinel is reachable from the begin iterator;
//   - TODO(hardening): both iterators refer to the same container.
//
// - `_LIBCPP_ASSERT_VALID_ELEMENT_ACCESS` -- checks that any attempts to access a container element, whether through
//   the container object or through an iterator, are valid and do not attempt to go out of bounds or otherwise access
//   a non-existent element. For iterator checks to work, bounded iterators must be enabled in the ABI. Types like
//   `optional` and `function` are considered one-element containers for the purposes of this check.
//
// - `_LIBCPP_ASSERT_NON_NULL` -- checks that the pointer being dereferenced is not null. On most modern platforms zero
//   address does not refer to an actual location in memory, so a null pointer dereference would not compromize the
//   memory security of a program (however, it is still undefined behavior that can result in strange errors due to
//   compiler optimizations).
//
// - `_LIBCPP_ASSERT_NON_OVERLAPPING_RANGES` -- for functions that take several ranges as arguments, checks that the
//   given ranges do not overlap.
//
// - `_LIBCPP_ASSERT_VALID_DEALLOCATION` -- checks that an attempt to deallocate memory is valid (e.g. the given object
//   was allocated by the given allocator). Violating this category typically results in a memory leak.
//
// - `_LIBCPP_ASSERT_VALID_EXTERNAL_API_CALL` -- checks that a call to an external API doesn't fail in
//   an unexpected manner. This includes triggering documented cases of undefined behavior in an external library (like
//   attempting to unlock an unlocked mutex in pthreads). Any API external to the library falls under this category
//   (from system calls to compiler intrinsics). We generally don't expect these failures to compromize memory safety or
//   otherwise create an immediate security issue.
//
// - `_LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR` -- checks any operations that exchange nodes between containers to make sure
//   the containers have compatible allocators.
//
// - `_LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN` -- checks that the given argument is within the domain of valid arguments
//   for the function. Violating this typically produces an incorrect result (e.g. the clamp algorithm returns the
//   original value without clamping it due to incorrect functors) or puts an object into an invalid state (e.g.
//   a string view where only a subset of elements is possible to access). This category is for assertions violating
//   which doesn't cause any immediate issues in the library -- whatever the consequences are, they will happen in the
//   user code.
//
// - `_LIBCPP_ASSERT_PEDANTIC` -- checks prerequisites which are imposed by the Standard, but violating which happens to
//   be benign in our implementation.
//
// - `_LIBCPP_ASSERT_SEMANTIC_REQUIREMENT` -- checks that the given argument satisfies the semantic requirements imposed
//   by the Standard. Typically, there is no simple way to completely prove that a semantic requirement is satisfied;
//   thus, this would often be a heuristic check and it might be quite expensive.
//
// - `_LIBCPP_ASSERT_INTERNAL` -- checks that internal invariants of the library hold. These assertions don't depend on
//   user input.
//
// - `_LIBCPP_ASSERT_UNCATEGORIZED` -- for assertions that haven't been properly classified yet.

// clang-format off
#  define _LIBCPP_HARDENING_MODE_NONE      (1 << 1)
#  define _LIBCPP_HARDENING_MODE_FAST      (1 << 2)
#  define _LIBCPP_HARDENING_MODE_EXTENSIVE (1 << 4) // Deliberately not ordered.
#  define _LIBCPP_HARDENING_MODE_DEBUG     (1 << 3)
// clang-format on

#ifndef _LIBCPP_HARDENING_MODE

#  ifndef _LIBCPP_HARDENING_MODE_DEFAULT
#    error _LIBCPP_HARDENING_MODE_DEFAULT is not defined. This definition should be set at configuration time in the \
`__config_site` header, please make sure your installation of libc++ is not broken.
#  endif

#  define _LIBCPP_HARDENING_MODE _LIBCPP_HARDENING_MODE_DEFAULT
#endif

#if _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_NONE && _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_FAST &&  \
    _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_EXTENSIVE &&                                                      \
    _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_DEBUG
#  error _LIBCPP_HARDENING_MODE must be set to one of the following values: \
_LIBCPP_HARDENING_MODE_NONE, \
_LIBCPP_HARDENING_MODE_FAST, \
_LIBCPP_HARDENING_MODE_EXTENSIVE, \
_LIBCPP_HARDENING_MODE_DEBUG
#endif

// The library provides the macro `_LIBCPP_ASSERTION_SEMANTIC` for configuring the assertion semantic used by hardening;
// it can be set to one of the following values:
//
// - `_LIBCPP_ASSERTION_SEMANTIC_IGNORE`;
// - `_LIBCPP_ASSERTION_SEMANTIC_OBSERVE`;
// - `_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE`;
// - `_LIBCPP_ASSERTION_SEMANTIC_ENFORCE`.
//
// libc++ assertion semantics generally mirror the evaluation semantics of C++26 Contracts:
// - `ignore` evaluates the assertion but doesn't do anything if it fails (note that it differs from the Contracts
//   `ignore` semantic which wouldn't evaluate the assertion at all);
// - `observe` logs an error (indicating, if possible, that the error is fatal) and continues execution;
// - `quick-enforce` terminates the program as fast as possible (via trapping);
// - `enforce` logs an error and then terminates the program.
//
// Additionally, a special `hardening-dependent` value selects the assertion semantic based on the hardening mode in
// effect: the production-capable modes (`fast` and `extensive`) map to `quick_enforce` and the `debug` mode maps to
// `enforce`. The `hardening-dependent` semantic cannot be selected explicitly, it is only used when no assertion
// semantic is provided by the user _and_ the library's default semantic is configured to be dependent on hardening.
//
// Notes:
// - Continuing execution after a hardening check fails results in undefined behavior; the `observe` semantic is meant
//   to make adopting hardening easier but should not be used outside of this scenario;
// - C++26 wording for Library Hardening precludes a conforming Hardened implementation from using the Contracts
//   `ignore` semantic when evaluating hardened preconditions in the Library. Libc++ allows using this semantic for
//   hardened preconditions, however, be aware that using `ignore` does not produce a conforming "Hardened"
//   implementation, unlike the other semantics above.
// clang-format off
#  define _LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT (1 << 1)
#  define _LIBCPP_ASSERTION_SEMANTIC_IGNORE              (1 << 2)
#  define _LIBCPP_ASSERTION_SEMANTIC_OBSERVE             (1 << 3)
#  define _LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE       (1 << 4)
#  define _LIBCPP_ASSERTION_SEMANTIC_ENFORCE             (1 << 5)
// clang-format on

// If the user attempts to configure the assertion semantic, check that it is allowed in the current environment.
#if defined(_LIBCPP_ASSERTION_SEMANTIC)
#  if !_LIBCPP_HAS_EXPERIMENTAL_LIBRARY
#    error "Assertion semantics are an experimental feature."
#  endif
#  if defined(_LIBCPP_CXX03_LANG)
#    error "Assertion semantics are not available in the C++03 mode."
#  endif
#endif // defined(_LIBCPP_ASSERTION_SEMANTIC)

// User-provided semantic takes top priority -- don't override if set.
#ifndef _LIBCPP_ASSERTION_SEMANTIC

#  ifndef _LIBCPP_ASSERTION_SEMANTIC_DEFAULT
#    error _LIBCPP_ASSERTION_SEMANTIC_DEFAULT is not defined. This definition should be set at configuration time in \
the `__config_site` header, please make sure your installation of libc++ is not broken.
#  endif

#  if _LIBCPP_ASSERTION_SEMANTIC_DEFAULT != _LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT
#    define _LIBCPP_ASSERTION_SEMANTIC _LIBCPP_ASSERTION_SEMANTIC_DEFAULT
#  else
#    if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
#      define _LIBCPP_ASSERTION_SEMANTIC _LIBCPP_ASSERTION_SEMANTIC_ENFORCE
#    else
#      define _LIBCPP_ASSERTION_SEMANTIC _LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE
#    endif
#  endif // _LIBCPP_ASSERTION_SEMANTIC_DEFAULT != _LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT

#endif // #ifndef _LIBCPP_ASSERTION_SEMANTIC

// Finally, validate the selected semantic (in case the user tries setting it to an incorrect value):
#if _LIBCPP_ASSERTION_SEMANTIC != _LIBCPP_ASSERTION_SEMANTIC_IGNORE &&                                                 \
    _LIBCPP_ASSERTION_SEMANTIC != _LIBCPP_ASSERTION_SEMANTIC_OBSERVE &&                                                \
    _LIBCPP_ASSERTION_SEMANTIC != _LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE &&                                          \
    _LIBCPP_ASSERTION_SEMANTIC != _LIBCPP_ASSERTION_SEMANTIC_ENFORCE
#  error _LIBCPP_ASSERTION_SEMANTIC must be set to one of the following values: \
_LIBCPP_ASSERTION_SEMANTIC_IGNORE, \
_LIBCPP_ASSERTION_SEMANTIC_OBSERVE, \
_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE, \
_LIBCPP_ASSERTION_SEMANTIC_ENFORCE
#endif

#endif // _LIBCPP___CONFIGURATION_HARDENING_H
