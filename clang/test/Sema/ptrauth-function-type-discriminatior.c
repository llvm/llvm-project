// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -verify -xobjective-c -fblocks
// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -verify -xc
// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -verify -xc++

// expected-no-diagnostics

#define discm(x) __builtin_ptrauth_type_discriminator(x)

struct Complete {};
struct Incomplete;

#ifndef __cplusplus
enum EIncomplete;
#endif

enum EComplete { enumerator };

_Static_assert(discm(void(void)) == 18983, "");
_Static_assert(discm(void()) == discm(void(void)), "");
_Static_assert(discm(void(int *)) == discm(void(float *)), "");
_Static_assert(discm(void(int *)) == discm(void(struct Incomplete *)), "");
_Static_assert(discm(void(struct Complete *)) == discm(void(struct Incomplete *)), "");
_Static_assert(discm(void(int *)) != discm(void(int)), "");
_Static_assert(discm(void(int)) != discm(void(int, ...)), "");
_Static_assert(discm(_Atomic(int *)()) == discm(int *()), "");
#ifndef __cplusplus
_Static_assert(discm(enum EIncomplete()) == discm(int()), "");
#endif
_Static_assert(discm(enum EComplete()) == discm(int()), "");
_Static_assert(discm(unsigned long()) == discm(int()), "");
_Static_assert(discm(char()) == discm(int()), "");
_Static_assert(discm(int(int (*)[10])) == discm(int(int (*)[9])), "");
_Static_assert(discm(void (int[10])) == discm(void (int *)), "");
_Static_assert(discm(void (int[*])) == discm(void (int *)), "");
_Static_assert(discm(void (void ())) == discm(void (void (*))), "");

#ifndef __cplusplus
typedef struct {} foo;
struct foo {};
_Static_assert(discm(void(foo)) == discm(void(struct foo)), "");
#endif

#ifdef __OBJC__
@interface I @end
_Static_assert(discm(id()) == discm(I*()), "");
_Static_assert(discm(id()) == discm(void*()), "");
_Static_assert(discm(id()) == discm(Class()), "");
_Static_assert(discm(void(^())()) == discm(id()), "");
#endif

#ifdef __cplusplus
_Static_assert(discm(void(Complete &)) != discm(void(Complete *)), "");
_Static_assert(discm(void(Complete &)) != discm(void(Complete &&)), "");
_Static_assert(discm(void(Incomplete &)) != discm(void(Incomplete &&)), "");
/* Descend into array and function types when using references. */
_Static_assert(discm(void(void (&)())) != discm(void (void (&)(int))), "");
_Static_assert(discm(void(void (&)())) != discm(void (int (&)())), "");
_Static_assert(discm(void(int (&)[10])) == discm(void(int (&)[9])), "");
_Static_assert(discm(void(int (&)[10])) == discm(void(int (&)[])), "");
_Static_assert(discm(void(int (&)[10])) != discm(void(float (&)[10])), "");
#endif

typedef __attribute__((ext_vector_type(4))) float vec4;
typedef __attribute__((ext_vector_type(16))) char char_vec16;
_Static_assert(discm(void (vec4)) == discm(void (char_vec16)), "");
