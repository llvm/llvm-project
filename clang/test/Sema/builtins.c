// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wstrlcpy-strlcat-size -Wno-string-plus-int -triple=i686-apple-darwin9
// This test needs to set the target because it uses __builtin_ia32_vec_ext_v4si

int test1(float a, int b) {
  return __builtin_isless(a, b); // expected-note {{declared here}}
}
int test2(int a, int b) {
  return __builtin_islessequal(a, b);  // expected-error {{floating point type}}
}

int test3(double a, float b) {
  return __builtin_isless(a, b);
}
int test4(int* a, double b) {
  return __builtin_islessequal(a, b);  // expected-error {{floating point type}}
}

int test5(float a, long double b) {
  return __builtin_isless(a, b, b);  // expected-error {{too many arguments}}
}
int test6(float a, long double b) {
  return __builtin_islessequal(a);  // expected-error {{too few arguments}}
}


#define CFSTR __builtin___CFStringMakeConstantString
void test7() {
  const void *X;
  X = CFSTR("\242"); // expected-warning {{input conversion stopped}}
  X = CFSTR("\0"); // no-warning
  X = CFSTR(242); // expected-error {{CFString literal is not a string constant}} expected-warning {{incompatible integer to pointer conversion}}
  X = CFSTR("foo", "bar"); // expected-error {{too many arguments to function call}}
}


// atomics.

void test9(short v) {
  unsigned i, old;

  old = __sync_fetch_and_add();  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add(&old);  // expected-error {{too few arguments to function call}}
  old = __sync_fetch_and_add((unsigned*)0, 42i); // expected-warning {{imaginary constants are a GNU extension}}

  // PR7600: Pointers are implicitly casted to integers and back.
  void *old_ptr = __sync_val_compare_and_swap((void**)0, 0, 0);

  // Ensure the return type is correct even when implicit casts are stripped
  // away. This triggers an assertion while checking the comparison otherwise.
  if (__sync_fetch_and_add(&old, 1) == 1) {
  }
}

// overloaded atomics should be declared only once.
void test9_1(volatile int* ptr, int val) {
  __sync_fetch_and_add_4(ptr, val);
}
void test9_2(volatile int* ptr, int val) {
  __sync_fetch_and_add(ptr, val);
}
void test9_3(volatile int* ptr, int val) {
  __sync_fetch_and_add_4(ptr, val);
  __sync_fetch_and_add(ptr, val);
  __sync_fetch_and_add(ptr, val);
  __sync_fetch_and_add_4(ptr, val);
  __sync_fetch_and_add_4(ptr, val);
}

void test9_4(volatile int* ptr, int val) {
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_fetch_and_nand(ptr, val);
}

// rdar://7236819
void test10(void) __attribute__((noreturn));

void test10(void) {
  __asm__("int3");
  __builtin_unreachable();

  // No warning about falling off the end of a noreturn function.
}

void test11(int X) {
  switch (X) {
  case __builtin_eh_return_data_regno(0):  // constant foldable.
    break;
  }

  __builtin_eh_return_data_regno(X);  // expected-error {{argument to '__builtin_eh_return_data_regno' must be a constant integer}}
}

// PR5062
void test12(void) __attribute__((__noreturn__));
void test12(void) {
  __builtin_trap();  // no warning because trap is noreturn.
}

void test_unknown_builtin(int a, int b) {
  __builtin_isles(a, b); // expected-error{{use of unknown builtin}} \
                         // expected-note{{did you mean '__builtin_isless'?}}
}

int test13() {
  __builtin_eh_return(0, 0); // no warning, eh_return never returns.
}

// <rdar://problem/8228293>
void test14() {
  int old;
  old = __sync_fetch_and_min((volatile int *)&old, 1);
}

// <rdar://problem/8336581>
void test15(const char *s) {
  __builtin_printf("string is %s\n", s);
}

// PR7885
int test16() {
  return __builtin_constant_p() + // expected-error{{too few arguments}}
         __builtin_constant_p(1, 2); // expected-error {{too many arguments}}
}

// __builtin_constant_p cannot resolve non-constants as a file scoped array.
int expr;
char y[__builtin_constant_p(expr) ? -1 : 1]; // no warning, the builtin is false.

// no warning, the builtin is false.
struct foo { int a; };
struct foo x = (struct foo) { __builtin_constant_p(42) ? 37 : 927 };

const int test17_n = 0;
const char test17_c[] = {1, 2, 3, 0};
const char test17_d[] = {1, 2, 3, 4};
typedef int __attribute__((vector_size(16))) IntVector;
struct Aggregate { int n; char c; };
enum Enum { EnumValue1, EnumValue2 };

typedef __typeof(sizeof(int)) size_t;
size_t strlen(const char *);

void test17() {
#define ASSERT(...) { enum { folded = (__VA_ARGS__) }; int arr[folded ? 1 : -1]; }
#define T(...) ASSERT(__builtin_constant_p(__VA_ARGS__))
#define F(...) ASSERT(!__builtin_constant_p(__VA_ARGS__))

  // __builtin_constant_p returns 1 if the argument folds to:
  //  - an arithmetic constant with value which is known at compile time
  T(test17_n);
  T(&test17_c[3] - test17_c);
  T(3i + 5); // expected-warning {{imaginary constant}}
  T(4.2 * 7.6);
  T(EnumValue1);
  T((enum Enum)(int)EnumValue2);

  //  - the address of the first character of a string literal, losslessly cast
  //    to any type
  T("string literal");
  T((double*)"string literal");
  T("string literal" + 0);
  T((long)"string literal");

  // ... and otherwise returns 0.
  F("string literal" + 1);
  F(&test17_n);
  F(test17_c);
  F(&test17_c);
  F(&test17_d);
  F((struct Aggregate){0, 1});
  F((IntVector){0, 1, 2, 3});
  F(test17);

  // Ensure that a technique used in glibc is handled correctly.
#define OPT(...) (__builtin_constant_p(__VA_ARGS__) && strlen(__VA_ARGS__) < 4)
  // FIXME: These are incorrectly treated as ICEs because strlen is treated as
  // a builtin.
  ASSERT(OPT("abc"));
  ASSERT(!OPT("abcd"));
  // In these cases, the strlen is non-constant, but the __builtin_constant_p
  // is 0: the array size is not an ICE but is foldable.
  ASSERT(!OPT(test17_c));        // expected-warning {{folding}}
  ASSERT(!OPT(&test17_c[0]));    // expected-warning {{folding}}
  ASSERT(!OPT((char*)test17_c)); // expected-warning {{folding}}
  ASSERT(!OPT(test17_d));        // expected-warning {{folding}}
  ASSERT(!OPT(&test17_d[0]));    // expected-warning {{folding}}
  ASSERT(!OPT((char*)test17_d)); // expected-warning {{folding}}

#undef OPT
#undef T
#undef F
}

void test18() {
  char src[1024];
  char dst[2048];
  size_t result;
  void *ptr;

  ptr = __builtin___memccpy_chk(dst, src, '\037', sizeof(src), sizeof(dst));
  result = __builtin___strlcpy_chk(dst, src, sizeof(dst), sizeof(dst));
  result = __builtin___strlcat_chk(dst, src, sizeof(dst), sizeof(dst));

  ptr = __builtin___memccpy_chk(dst, src, '\037', sizeof(src));      // expected-error {{too few arguments to function call}}
  ptr = __builtin___strlcpy_chk(dst, src, sizeof(dst), sizeof(dst)); // expected-warning {{incompatible integer to pointer conversion}}
  ptr = __builtin___strlcat_chk(dst, src, sizeof(dst), sizeof(dst)); // expected-warning {{incompatible integer to pointer conversion}}
}

void no_ms_builtins() {
  __assume(1); // expected-warning {{implicit declaration}}
  __noop(1); // expected-warning {{implicit declaration}}
  __debugbreak(); // expected-warning {{implicit declaration}}
}

void unavailable() {
  __builtin_operator_new(0); // expected-error {{'__builtin_operator_new' is only available in C++}}
  __builtin_operator_delete(0); // expected-error {{'__builtin_operator_delete' is only available in C++}}
}

// rdar://18259539
size_t strlcpy(char * restrict dst, const char * restrict src, size_t size);
size_t strlcat(char * restrict dst, const char * restrict src, size_t size);

void Test19(void)
{
        static char b[40];
        static char buf[20];

        strlcpy(buf, b, sizeof(b)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} \\
                                    // expected-note {{change size argument to be the size of the destination}}
        __builtin___strlcpy_chk(buf, b, sizeof(b), __builtin_object_size(buf, 0)); // expected-warning {{size argument in '__builtin___strlcpy_chk' call appears to be size of the source; expected the size of the destination}} \
                                    // expected-note {{change size argument to be the size of the destination}} \
				    // expected-warning {{'strlcpy' will always overflow; destination buffer has size 20, but size argument is 40}}

        strlcat(buf, b, sizeof(b)); // expected-warning {{size argument in 'strlcat' call appears to be size of the source; expected the size of the destination}} \
                                    // expected-note {{change size argument to be the size of the destination}}
				    
        __builtin___strlcat_chk(buf, b, sizeof(b), __builtin_object_size(buf, 0)); // expected-warning {{size argument in '__builtin___strlcat_chk' call appears to be size of the source; expected the size of the destination}} \
                                                                                   // expected-note {{change size argument to be the size of the destination}} \
				                                                   // expected-warning {{'strlcat' will always overflow; destination buffer has size 20, but size argument is 40}}
}

// rdar://11076881
char * Test20(char *p, const char *in, unsigned n)
{
    static char buf[10];

    __builtin___memcpy_chk (&buf[6], in, 5, __builtin_object_size (&buf[6], 0)); // expected-warning {{'memcpy' will always overflow; destination buffer has size 4, but size argument is 5}}

    __builtin___memcpy_chk (p, "abcde", n, __builtin_object_size (p, 0));

    __builtin___memcpy_chk (&buf[5], "abcde", 5, __builtin_object_size (&buf[5], 0));

    __builtin___memcpy_chk (&buf[5], "abcde", n, __builtin_object_size (&buf[5], 0));

    __builtin___memcpy_chk (&buf[6], "abcde", 5, __builtin_object_size (&buf[6], 0)); // expected-warning {{'memcpy' will always overflow; destination buffer has size 4, but size argument is 5}}

    return buf;
}

typedef void (fn_t)(int);

void test_builtin_launder(char *p, void *vp, const void *cvp,
                          const volatile int *ip, float *restrict fp,
                          fn_t *fn) {
  __builtin_launder(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_launder(p, p); // expected-error {{too many arguments to function call, expected 1, have 2}}
  int x;
  __builtin_launder(x); // expected-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  char *d = __builtin_launder(p);
  __builtin_launder(vp);  // expected-error {{void pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(cvp); // expected-error {{void pointer argument to '__builtin_launder' is not allowed}}
  const volatile int *id = __builtin_launder(ip);
  int *id2 = __builtin_launder(ip); // expected-warning {{discards qualifiers}}
  float *fd = __builtin_launder(fp);
  __builtin_launder(fn); // expected-error {{function pointer argument to '__builtin_launder' is not allowed}}
}

void test21(const int *ptr) {
  __sync_fetch_and_add(ptr, 1); // expected-error{{address argument to atomic builtin cannot be const-qualified ('const int *' invalid)}}
  __atomic_fetch_add(ptr, 1, 0);  // expected-error {{address argument to atomic operation must be a pointer to non-const type ('const int *' invalid)}}
}

void test_ei_i42i(_ExtInt(42) *ptr, int value) {
  __sync_fetch_and_add(ptr, value); // expected-error {{Atomic memory operand must have a power-of-two size}}
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expected-error {{Atomic memory operand must have a power-of-two size}}

  __atomic_fetch_add(ptr, 1, 0); // expected-error {{argument to atomic builtin of type '_ExtInt' is not supported}}
}

void test_ei_i64i(_ExtInt(64) *ptr, int value) {
  __sync_fetch_and_add(ptr, value); // expect success
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expect success

  __atomic_fetch_add(ptr, 1, 0); // expected-error {{argument to atomic builtin of type '_ExtInt' is not supported}}
}

void test_ei_ii42(int *ptr, _ExtInt(42) value) {
  __sync_fetch_and_add(ptr, value); // expect success
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expect success
}

void test_ei_ii64(int *ptr, _ExtInt(64) value) {
  __sync_fetch_and_add(ptr, value); // expect success
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expect success
}

void test_ei_i42i42(_ExtInt(42) *ptr, _ExtInt(42) value) {
  __sync_fetch_and_add(ptr, value); // expected-error {{Atomic memory operand must have a power-of-two size}}
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expected-error {{Atomic memory operand must have a power-of-two size}}
}

void test_ei_i64i64(_ExtInt(64) *ptr, _ExtInt(64) value) {
  __sync_fetch_and_add(ptr, value); // expect success
  // expected-warning@+1 {{the semantics of this intrinsic changed with GCC version 4.4 - the newer semantics are provided here}}
  __sync_nand_and_fetch(ptr, value); // expect success
}

void test22(void) {
  (void)__builtin_signbit(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  (void)__builtin_signbit(1.0, 2.0, 3.0); // expected-error{{too many arguments to function call, expected 1, have 3}}
  (void)__builtin_signbit(1); // expected-error {{floating point classification requires argument of floating point type (passed in 'int')}}
  (void)__builtin_signbit(1.0);
  (void)__builtin_signbit(1.0f);
  (void)__builtin_signbit(1.0L);

  (void)__builtin_signbitf(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  (void)__builtin_signbitf(1.0, 2.0, 3.0); // expected-error{{too many arguments to function call, expected 1, have 3}}
  (void)__builtin_signbitf(1);
  (void)__builtin_signbitf(1.0);
  (void)__builtin_signbitf(1.0f);
  (void)__builtin_signbitf(1.0L);

  (void)__builtin_signbitl(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  (void)__builtin_signbitl(1.0, 2.0, 3.0); // expected-error{{too many arguments to function call, expected 1, have 3}}
  (void)__builtin_signbitl(1);
  (void)__builtin_signbitl(1.0);
  (void)__builtin_signbitl(1.0f);
  (void)__builtin_signbitl(1.0L);
}

// rdar://43909200
#define memcpy(x,y,z) __builtin___memcpy_chk(x,y,z, __builtin_object_size(x,0))
#define my_memcpy(x,y,z) __builtin___memcpy_chk(x,y,z, __builtin_object_size(x,0))

void test23() {
  char src[1024];
  char buf[10];
  memcpy(buf, src, 11); // expected-warning{{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
  my_memcpy(buf, src, 11); // expected-warning{{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

// Test that __builtin_is_constant_evaluated() is not allowed in C
int test_cxx_builtin() {
  // expected-error@+1 {{use of unknown builtin '__builtin_is_constant_evaluated'}}
  return __builtin_is_constant_evaluated();
}

void test_builtin_complex() {
  __builtin_complex(); // expected-error {{too few}}
  __builtin_complex(1); // expected-error {{too few}}
  __builtin_complex(1, 2, 3); // expected-error {{too many}}

  _Static_assert(_Generic(__builtin_complex(1.0f, 2.0f), _Complex float: 1, default: 0), "");
  _Static_assert(_Generic(__builtin_complex(1.0, 2.0), _Complex double: 1, default: 0), "");
  _Static_assert(_Generic(__builtin_complex(1.0l, 2.0l), _Complex long double: 1, default: 0), "");

  __builtin_complex(1, 2); // expected-error {{argument type 'int' is not a real floating point type}}
  __builtin_complex(1, 2.0); // expected-error {{argument type 'int' is not a real floating point type}}
  __builtin_complex(1.0, 2); // expected-error {{argument type 'int' is not a real floating point type}}

  __builtin_complex(1.0, 2.0f); // expected-error {{arguments are of different types ('double' vs 'float')}}
  __builtin_complex(1.0f, 2.0); // expected-error {{arguments are of different types ('float' vs 'double')}}
}

_Complex double builtin_complex_static_init = __builtin_complex(1.0, 2.0);
