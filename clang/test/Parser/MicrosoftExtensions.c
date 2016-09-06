// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -Wno-missing-declarations -verify -fms-extensions  %s
__stdcall int func0(void);
int __stdcall func(void);
typedef int (__cdecl *tptr)(void);
void (*__fastcall fastpfunc)(void);
extern __declspec(dllimport) void __stdcall VarR4FromDec(void);
__declspec(deprecated) __declspec(deprecated) char * __cdecl ltoa( long _Val, char * _DstBuf, int _Radix);
__declspec(safebuffers) __declspec(noalias) __declspec(restrict) void * __cdecl xxx(void *_Memory); /* expected-warning{{__declspec attribute 'safebuffers' is not supported}} */
typedef __w64 unsigned long ULONG_PTR, *PULONG_PTR;

void * __ptr64 PtrToPtr64(const void *p) {
  return((void * __ptr64) (unsigned __int64) (ULONG_PTR)p );
}

void * __ptr32 PtrToPtr32(const void *p) {
  return((void * __ptr32) (unsigned __int32) (ULONG_PTR)p );
}

/* Both inline and __forceinline is OK. */
inline void __forceinline pr8264(void) {}
__forceinline void inline pr8264_1(void) {}
void inline __forceinline pr8264_2(void) {}
void __forceinline inline pr8264_3(void) {}
/* But duplicate __forceinline causes warning. */
void __forceinline __forceinline pr8264_4(void) {  /* expected-warning{{duplicate '__forceinline' declaration specifier}} */
}

_inline int foo99(void) { return 99; }

void test_ms_alignof_alias(void) {
  unsigned int s = _alignof(int);
  s = __builtin_alignof(int);
}

/* Charify extension. */
#define FOO(x) #@x
char x = FOO(a);
#define HASHAT #@
#define MISSING_ARG(x) #@
/* expected-error@-1 {{'#@' is not followed by a macro parameter}} */

typedef enum E { e1 };

enum __declspec(deprecated) E2 { i, j, k }; /* expected-note {{'E2' has been explicitly marked deprecated here}} */
__declspec(deprecated) enum E3 { a, b, c } e; /* expected-note {{'e' has been explicitly marked deprecated here}} */

void deprecated_enum_test(void) {
  /* Test to make sure the deprecated warning follows the right thing */
  enum E2 e1;  /* expected-warning {{'E2' is deprecated}} */
  enum E3 e2; /* No warning expected, the deprecation follows the variable */
  enum E3 e3 = e;  /* expected-warning {{'e' is deprecated}} */
}

/* Microsoft attribute tests */
[returnvalue:SA_Post( attr=1)]
int foo1([SA_Post(attr=1)] void *param);

[unbalanced(attribute) /* expected-note {{to match this '['}} */
void f(void); /* expected-error {{expected ']'}} */

void ms_intrinsics(int a) {
  __noop();
  __assume(a);
  __debugbreak();
}

struct __declspec(frobble) S1 {};	/* expected-warning {{__declspec attribute 'frobble' is not supported}} */
struct __declspec(12) S2 {};	/* expected-error {{__declspec attributes must be an identifier or string literal}} */
struct __declspec("testing") S3 {}; /* expected-warning {{__declspec attribute '"testing"' is not supported}} */

/* declspecs with arguments cannot have an empty argument list, even if the
   arguments are optional. */
__declspec(deprecated()) void dep_func_test(void); /* expected-error {{parentheses must be omitted if 'deprecated' attribute's argument list is empty}} */
__declspec(deprecated) void dep_func_test2(void);
__declspec(deprecated("")) void dep_func_test3(void);

/* Ensure multiple declspec attributes are supported */
struct __declspec(align(8) deprecated) S4 {};

/* But multiple declspecs must still be legal */
struct __declspec(deprecated frobble "testing") S5 {};  /* expected-warning {{__declspec attribute 'frobble' is not supported}} expected-warning {{__declspec attribute '"testing"' is not supported}} */
struct __declspec(unknown(12) deprecated) S6 {};	/* expected-warning {{__declspec attribute 'unknown' is not supported}}*/

int * __sptr psp;
int * __uptr pup;
/* Either ordering is acceptable */
int * __ptr32 __sptr psp32;
int * __ptr32 __uptr pup32;
int * __sptr __ptr64 psp64;
int * __uptr __ptr64 pup64;

/* Legal to have nested pointer attributes */
int * __sptr * __ptr32 ppsp32;

// Ignored type qualifiers after comma in declarator lists
typedef int ignored_quals_dummy1, const volatile __ptr32 __ptr64 __w64 __unaligned __sptr __uptr ignored_quals1; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy2)(), __fastcall ignored_quals2; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy3)(), __stdcall ignored_quals3; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy4)(), __thiscall ignored_quals4; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy5)(), __cdecl ignored_quals5; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy6)(), __vectorcall ignored_quals6; // expected-warning {{qualifiers after comma in declarator list are ignored}}

__declspec(align(16)) struct align_before_key1 {};
__declspec(align(16)) struct align_before_key2 {} align_before_key2_var;
__declspec(align(16)) struct align_before_key3 {} *align_before_key3_var;
_Static_assert(__alignof(struct align_before_key1) == 16, "");
_Static_assert(__alignof(struct align_before_key2) == 16, "");
_Static_assert(__alignof(struct align_before_key3) == 16, "");

void PR28782(int i) {
foo:
  int n;
  switch (i) {
  case 0:
    int m;
  }
}
