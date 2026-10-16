// RUN: %clang_cc1 -verify -std=c2y -DSTD1 %s
// RUN: %clang_cc1 -verify -std=c2y -DSTD2 %s
// RUN: %clang_cc1 -verify=gnu1 -std=gnu2y -DGNU1 %s
// RUN: %clang_cc1 -verify -std=gnu2y -DGNU2 %s
// RUN: %clang_cc1 -verify=gnu3 -std=gnu2y -DGNU3 %s
// RUN: %clang_cc1 -verify -std=gnu2y -DGNU4 %s
// RUN: %clang_cc1 -verify -std=gnu2y -DGNU5 %s
// RUN: %clang_cc1 -verify -std=gnu2y -DGNU6 %s
// RUN: %clang_cc1 -verify=gnu7 -std=gnu2y -DGNU7 %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -verify -std=c2y -DDARWIN1 %s
// RUN: %clang_cc1 -triple x86_64-pc-win32-mscv -verify -std=c2y -fms-compatibility -DMS1 %s
// RUN: %clang_cc1 -triple x86_64-pc-win32-mscv -verify -std=c2y -fms-compatibility -DMS2 %s
// RUN: %clang_cc1 -verify=invalid -std=c2y -DINVALID1 %s
// RUN: %clang_cc1 -verify=invalid -std=c2y -DINVALID2 %s
// expected-no-diagnostics

/* WG14 N3623: Yes
 * Earthly Demon XV: Definition of Main
 *
 * This validates that we accept the standard type definitions of main or some
 * other implementation-defined type.
 */

typedef __WCHAR_TYPE__ wchar_t;

// These are the signatures required by the standard.
#if defined(STD1)
int main(void) {}
#elif defined(STD2)
int main(int argc, char *argv[]) {}
#endif

// GNU extensions.
#if defined(GNU1)
void main(void) {} /* gnu1-warning {{return type of 'main' is not 'int'}}
                      gnu1-note {{change return type to 'int'}}
                    */
#elif defined(GNU2)
const int main(void) {}
#elif defined(GNU3)
int main(...) {} /* gnu3-warning {{'main' is not allowed to be declared variadic}} */
#elif defined(GNU4)
int main(int, const char **) {}
#elif defined(GNU5)
int main(int, char const * const *) {}
#elif defined(GNU6)
int main(int, char * const *) {}
#elif defined(GNU7)
int main(int) {} /* gnu7-warning {{only one parameter on 'main' declaration}} */
#endif

// Darwin extensions.
#if defined(DARWIN1)
int main(int argc, char *argv[], char *environ[], char **undocumented) {}
#endif

// Microsoft extensions.
#if defined(MS1)
int wmain(int, wchar_t *[]) {}
#elif defined(MS2)
int wmain(int, wchar_t *[], wchar_t *[]) {}
#endif

// Invalid signatures.
#if defined(INVALID1)
inline int main(int, char *[]); /* invalid-error {{'main' is not allowed to be declared inline}} */
#if !__is_target_os(darwin)
// This test doesn't make sense on Darwin where four arguments are allowed.
int main(int, char *[], char *[], float); /* invalid-error {{too many parameters (4) for 'main': must be 0, 2, or 3}} */
#endif
float main(int); /* invalid-error {{'main' must return 'int'}} */
_Noreturn int main(int, char *[]); /* invalid-warning {{'main' is not allowed to be declared _Noreturn}}
                                      invalid-note {{remove '_Noreturn'}}
                                    */
#elif defined(INVALID2)
static int main(void); /* invalid-warning {{'main' should not be declared static}} */
#endif
