// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=unlimited -ftype-print-limit=0 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=unlimited %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=limit6 -ftype-print-limit=6 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=limit3 -ftype-print-limit=3 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify=limit10 -ftype-print-limit=10 %s

// The number of template arguments (and function parameters) printed when
// spelling a type is limited. Once the limit is reached, the arguments that
// would open another list of arguments are replaced by a marker, while the ones
// that don't (builtin types, ...) are still printed. The default limit is large
// enough not to affect any of the types below.

template <class A, class B> struct F {
  struct Inner {};
};
using T0 = int;
using T1 = F<T0, T0>;
using T2 = F<T1, T1>;
using T3 = F<T2, T2>;
using T4 = F<T3, T3>;

// A mix of arguments that nest and ones that don't.
template <class... Ts> struct G {};
using M = G<int, F<int, int>, char, F<int, int>, F<int, int>>;

// The arguments are printed left to right, so the left of the type is where
// the limit is reached, and what is omitted is on the right. A single omitted
// argument says so in the singular.
int a = T4(); // unlimited-error {{no viable conversion from 'T4' (aka 'F<F<F<F<int, int>, F<int, int>>, F<F<int, int>, F<int, int>>>, F<F<F<int, int>, F<int, int>>, F<F<int, int>, F<int, int>>>>') to 'int'}}
              // limit6-error@-1 {{no viable conversion from 'T4' (aka 'F<F<F<F<int, int>, F<int, int>>, [arg omitted]>, [arg omitted]>') to 'int'}}
              // limit3-error@-2 {{no viable conversion from 'T4' (aka 'F<F<F<F<int, int>, [arg omitted]>, [arg omitted]>, [arg omitted]>') to 'int'}}
              // limit10-error@-3 {{no viable conversion from 'T4' (aka 'F<F<F<F<int, int>, F<int, int>>, F<F<int, int>, [arg omitted]>>, [arg omitted]>') to 'int'}}

// The arguments that don't nest (int, char) are printed regardless.
int b = M(); // unlimited-error {{no viable conversion from 'M' (aka 'G<int, F<int, int>, char, F<int, int>, F<int, int>>') to 'int'}}
             // limit6-error@-1 {{no viable conversion from 'M' (aka 'G<int, F<int, int>, char, F<int, int>, [arg omitted]>') to 'int'}}
             // limit3-error@-2 {{no viable conversion from 'M' (aka 'G<int, F<int, int>, char, [2 args omitted]>') to 'int'}}
             // limit10-error@-3 {{no viable conversion from 'M' (aka 'G<int, F<int, int>, char, F<int, int>, F<int, int>>') to 'int'}}

// Types that are not sugared, or small, are not affected.
int d = T4::Inner(); // unlimited-error {{no viable conversion from 'T4::Inner' to 'int'}}
                     // limit6-error@-1 {{no viable conversion from 'T4::Inner' to 'int'}}
                     // limit3-error@-2 {{no viable conversion from 'T4::Inner' to 'int'}}
                     // limit10-error@-3 {{no viable conversion from 'T4::Inner' to 'int'}}
int e = F<int, char>(); // unlimited-error {{no viable conversion from 'F<int, char>' to 'int'}}
                        // limit6-error@-1 {{no viable conversion from 'F<int, char>' to 'int'}}
                        // limit3-error@-2 {{no viable conversion from 'F<int, char>' to 'int'}}
                        // limit10-error@-3 {{no viable conversion from 'F<int, char>' to 'int'}}

// Function parameter lists count the same way.
using P0 = void (*)();
using P1 = void (*)(P0, P0);
using P2 = void (*)(P1, P1);
using P3 = void (*)(P2, P2);
int c = P3(); // unlimited-error {{with an rvalue of type 'P3' (aka 'void (*)(void (*)(void (*)(void (*)(), void (*)()), void (*)(void (*)(), void (*)())), void (*)(void (*)(void (*)(), void (*)()), void (*)(void (*)(), void (*)())))')}}
              // limit6-error@-1 {{with an rvalue of type 'P3' (aka 'void (*)(void (*)(void (*)(void (*)(), void (*)()), void (*)(void (*)(), [arg omitted])), [arg omitted])')}}
              // limit3-error@-2 {{with an rvalue of type 'P3' (aka 'void (*)(void (*)(void (*)(void (*)(), [arg omitted]), [arg omitted]), [arg omitted])')}}
              // limit10-error@-3 {{with an rvalue of type 'P3' (aka 'void (*)(void (*)(void (*)(void (*)(), void (*)()), void (*)(void (*)(), void (*)())), void (*)(void (*)(void (*)(), [arg omitted]), [arg omitted]))')}}
