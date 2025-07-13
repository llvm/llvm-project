/* RUN: %clang_cc1 -fsyntax-only -std=c2y -pedantic -Wpre-c2y-compat -verify=compat %s
   RUN: %clang_cc1 -fsyntax-only -std=c23 -pedantic -verify %s
   RUN: %clang_cc1 -fsyntax-only -std=c89 -pedantic -verify=expected,static-assert %s
   RUN: %clang_cc1 -fsyntax-only -pedantic -verify=cpp,static-assert -x c++ %s
 */

/* This tests the extension behavior for _Countof in language modes before C2y.
 * It also tests the behavior of the precompat warning. And it tests the
 * behavior in C++ mode where the extension is not supported.
 */

#if __STDC_VERSION__ < 202400L && !defined(__cplusplus)
#if __has_feature(c_countof)
#error "Did not expect _Countof to be a feature in older language modes"
#endif

#if !__has_extension(c_countof)
#error "Expected _Countof to be an extension in older language modes"
#endif
#endif /* __STDC_VERSION__ < 202400L && !defined(__cplusplus) */

#ifdef __cplusplus
/* C++ should not have this as a feature or as an extension. */
#if __has_feature(c_countof) || __has_extension(c_countof)
#error "did not expect there to be _Countof support"
#endif
#endif /* __cplusplus */

int array[12];
int x = _Countof(array);   /* expected-warning {{'_Countof' is a C2y extension}}
                              compat-warning {{'_Countof' is incompatible with C standards before C2y}}
                              cpp-error {{use of undeclared identifier '_Countof'}}
                            */
int y = _Countof(int[12]); /* expected-warning {{'_Countof' is a C2y extension}}
                              compat-warning {{'_Countof' is incompatible with C standards before C2y}}
                              cpp-error {{expected '(' for function-style cast or type construction}}
                            */

_Static_assert(_Countof(int[12]) == 12, ""); /* expected-warning {{'_Countof' is a C2y extension}}
                                                compat-warning {{'_Countof' is incompatible with C standards before C2y}}
                                                cpp-error {{expected '(' for function-style cast or type construction}}
                                                static-assert-warning {{'_Static_assert' is a C11 extension}}
                                              */
