// RUN: %clang_cc1 -fsyntax-only -Wdeprecated-declarations -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdeprecated-declarations -DUNRELATED_PRAGMA -verify=escape %s
// RUN: %clang_cc1 -fsyntax-only -Wdeprecated-declarations -DNO_PRAGMA -verify=escape %s

// A deprecation warning emitted inside a system header is shown when the
// instantiation is requested from user code, but a pragma suppressing it at
// the point of instantiation must silence it (GH219685). Pragmas for other
// groups, or no pragma at all, must not.

#ifdef BE_THE_HEADER
#pragma clang system_header

template <class T>
struct TUnderlying {
  using Type = __underlying_type(T); // escape-warning {{'EDoomed' is deprecated}} \
                                     // escape-warning {{'EGone' is deprecated: use EFine}}
};

#else
#define BE_THE_HEADER
#include __FILE__

enum class [[deprecated]] EDoomed { A }; // escape-note 2 {{'EDoomed' has been explicitly marked deprecated here}}
enum class [[deprecated("use EFine")]] EGone { A }; // escape-note 2 {{'EGone' has been explicitly marked deprecated here}}

#if defined(UNRELATED_PRAGMA)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif
#if !defined(UNRELATED_PRAGMA) && !defined(NO_PRAGMA)
// expected-no-diagnostics
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

using FUnder = TUnderlying<EDoomed>::Type; // escape-warning {{'EDoomed' is deprecated}} \
                                           // escape-note {{in instantiation of template class 'TUnderlying<EDoomed>' requested here}}
using FUnder2 = TUnderlying<EGone>::Type; // escape-warning {{'EGone' is deprecated: use EFine}} \
                                          // escape-note {{in instantiation of template class 'TUnderlying<EGone>' requested here}}

#if !defined(NO_PRAGMA)
#pragma clang diagnostic pop
#endif

#endif
