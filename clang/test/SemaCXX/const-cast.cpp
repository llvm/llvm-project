// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct A {};

// See if aliasing can confuse this baby.
typedef char c;
typedef c *cp;
typedef cp *cpp;
typedef cpp *cppp;
typedef cppp &cpppr;
typedef const cppp &cpppcr;
typedef const char cc;
typedef cc *ccp;
typedef volatile ccp ccvp;
typedef ccvp *ccvpp;
typedef const volatile ccvpp ccvpcvp;
typedef ccvpcvp *ccvpcvpp;
typedef int iar[100];
typedef iar &iarr;
typedef int (*f)(int);

char ***good_const_cast_test(ccvpcvpp var)
{
  // Cast away deep consts and volatiles.
  char ***var2 = const_cast<cppp>(var);
  char ***const &var3 = var2;
  // Const reference to reference.
  char ***&var4 = const_cast<cpppr>(var3);
  // Drop reference. Intentionally without qualifier change.
  char *** var5 = const_cast<cppp>(var4);
  // Const array to array reference.
  const int ar[100] = {0};
  int (&rar)[100] = const_cast<iarr>(ar);
  // Array decay. Intentionally without qualifier change.
  int *pi = const_cast<int*>(ar);
  f fp = 0;
  // Don't misidentify fn** as a function pointer.
  f *fpp = const_cast<f*>(&fp);
  int const A::* const A::*icapcap = 0;
  int A::* A::* iapap = const_cast<int A::* A::*>(icapcap);
  (void)const_cast<A&&>(A());
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2 {{rvalue references are a C++11 extension}}
#endif
  return var4;
}

short *bad_const_cast_test(char const *volatile *const volatile *var)
{
  // Different pointer levels.
  char **var2 = const_cast<char**>(var); // expected-error {{const_cast from 'const char *volatile *const volatile *' to 'char **' is not allowed}}
  // Different final type.
  short ***var3 = const_cast<short***>(var); // expected-error {{const_cast from 'const char *volatile *const volatile *' to 'short ***' is not allowed}}
  // Rvalue to reference.
  char ***&var4 = const_cast<cpppr>(&var2); // expected-error {{const_cast from rvalue to reference type 'cpppr'}}
  // Non-pointer.
  char v = const_cast<char>(**var2); // expected-error {{const_cast to 'char', which is not a reference, pointer-to-object, or pointer-to-data-member}}
  const int *ar[100] = {0};
  extern const int *aub[];
  // const_cast looks through arrays as of DR330.
  (void) const_cast<int *(*)[100]>(&ar); // ok
  (void) const_cast<int *(*)[]>(&aub); // ok
  // ... but the array bound must exactly match.
  (void) const_cast<int *(*)[]>(&ar); // expected-error {{const_cast from 'const int *(*)[100]' to 'int *(*)[]' is not allowed}}
  (void) const_cast<int *(*)[99]>(&ar); // expected-error {{const_cast from 'const int *(*)[100]' to 'int *(*)[99]' is not allowed}}
  (void) const_cast<int *(*)[100]>(&aub); // expected-error {{const_cast from 'const int *(*)[]' to 'int *(*)[100]' is not allowed}}
  f fp1 = 0;
  // Function pointers.
  f fp2 = const_cast<f>(fp1); // expected-error {{const_cast to 'f' (aka 'int (*)(int)'), which is not a reference, pointer-to-object, or pointer-to-data-member}}
  void (A::*mfn)() = 0;
  (void)const_cast<void (A::*)()>(mfn); // expected-error-re {{const_cast to 'void (A::*)(){{( __attribute__\(\(thiscall\)\))?}}', which is not a reference, pointer-to-object, or pointer-to-data-member}}
  (void)const_cast<int&&>(0); // expected-error {{const_cast from rvalue to reference type 'int &&'}}
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2 {{rvalue references are a C++11 extension}}
#endif
  return **var3;
}

template <typename T>
char *PR21845() { return const_cast<char *>((void)T::x); } // expected-error {{const_cast from 'void' to 'char *' is not allowed}}
