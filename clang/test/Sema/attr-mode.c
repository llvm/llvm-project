// RUN: %clang_cc1 -triple i686-pc-linux-gnu -DTEST_32BIT_X86 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -DTEST_64BIT_X86 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple powerpc64-pc-linux-gnu -DTEST_64BIT_PPC64 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple powerpc64-pc-linux-gnu -DTEST_F128_PPC64 -fsyntax-only \
// RUN:   -verify -target-feature +float128 %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnux32 -DTEST_64BIT_X86 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple mips-linux-gnu -DTEST_MIPS_32 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple mips64-linux-gnuabin32 -DTEST_MIPS_N32 -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -DTEST_MIPS_64 -fsyntax-only \
// RUN:   -verify %s

typedef int i16_1 __attribute((mode(HI)));
int i16_1_test[sizeof(i16_1) == 2 ? 1 : -1];
typedef int i16_2 __attribute((__mode__(__HI__)));
int i16_2_test[sizeof(i16_1) == 2 ? 1 : -1];

typedef float f64 __attribute((mode(DF)));
int f64_test[sizeof(f64) == 8 ? 1 : -1];

typedef int invalid_1 __attribute((mode)); // expected-error{{'mode' attribute takes one argument}}
typedef int invalid_2 __attribute((mode())); // expected-error{{'mode' attribute takes one argument}}
typedef int invalid_3 __attribute((mode(II))); // expected-error{{unknown machine mode}}
typedef struct {int i,j,k;} invalid_4 __attribute((mode(SI))); // expected-error{{mode attribute only supported for integer and floating-point types}}
typedef float invalid_5 __attribute((mode(SI))); // expected-error{{type of machine mode does not match type of base type}}
typedef int invalid_6 __attribute__((mode(12)));  // expected-error{{'mode' attribute requires an identifier}}

typedef unsigned unwind_word __attribute((mode(unwind_word)));

int **__attribute((mode(QI)))* i32;  // expected-error{{mode attribute}}

__attribute__((mode(QI))) int invalid_func(void) { return 1; } // expected-error{{'mode' attribute only applies to variables, enums, typedefs, and non-static data members}}
enum invalid_enum { A1 __attribute__((mode(QI))) }; // expected-error{{'mode' attribute only applies to}}

typedef _Complex double c32 __attribute((mode(SC)));
int c32_test[sizeof(c32) == 8 ? 1 : -1];
typedef _Complex float c64 __attribute((mode(DC)));

#if !defined(__ppc__) && !defined(__mips__) // Note, 'XC' mode is illegal for PPC64 and MIPS machines.
typedef _Complex float c80 __attribute((mode(XC)));
#endif

// PR6108: Correctly select 'long' built in type on 64-bit platforms for 64 bit
// modes. Also test other mode-based conversions.
typedef int i8_mode_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int ui8_mode_t __attribute__ ((__mode__ (__QI__)));
typedef int i16_mode_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int ui16_mode_t __attribute__ ((__mode__ (__HI__)));
typedef int i32_mode_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int ui32_mode_t __attribute__ ((__mode__ (__SI__)));
typedef int i64_mode_t __attribute__ ((__mode__ (__DI__)));
typedef unsigned int ui64_mode_t __attribute__ ((__mode__ (__DI__)));
void f_i8_arg(i8_mode_t* x) { (void)x; }
void f_ui8_arg(ui8_mode_t* x) { (void)x; }
void f_i16_arg(i16_mode_t* x) { (void)x; }
void f_ui16_arg(ui16_mode_t* x) { (void)x; }
void f_i32_arg(i32_mode_t* x) { (void)x; }
void f_ui32_arg(ui32_mode_t* x) { (void)x; }
void f_i64_arg(i64_mode_t* x) { (void)x; }
void f_ui64_arg(ui64_mode_t* x) { (void)x; }
void test_char_to_i8(signed char* y) { f_i8_arg(y); }
void test_char_to_ui8(unsigned char* y) { f_ui8_arg(y); }
void test_short_to_i16(short* y) { f_i16_arg(y); }
void test_short_to_ui16(unsigned short* y) { f_ui16_arg(y); }
void test_int_to_i32(int* y) { f_i32_arg(y); }
void test_int_to_ui32(unsigned int* y) { f_ui32_arg(y); }
#if TEST_32BIT_X86
void test_long_to_i64(long long* y) { f_i64_arg(y); }
void test_long_to_ui64(unsigned long long* y) { f_ui64_arg(y); }
#elif TEST_64BIT_X86
#ifdef __ILP32__
typedef unsigned int gcc_word __attribute__((mode(word)));
int foo[sizeof(gcc_word) == 8 ? 1 : -1];
typedef unsigned int gcc_unwind_word __attribute__((mode(unwind_word)));
int foo[sizeof(gcc_unwind_word) == 8 ? 1 : -1];
void test_long_to_i64(long long* y) { f_i64_arg(y); }
void test_long_to_ui64(unsigned long long* y) { f_ui64_arg(y); }
#else
void test_long_to_i64(long* y) { f_i64_arg(y); }
void test_long_to_ui64(unsigned long* y) { f_ui64_arg(y); }
#endif
typedef          float f128ibm __attribute__ ((mode (TF)));
#elif TEST_64BIT_PPC64
typedef          float f128ibm __attribute__ ((mode (TF)));
typedef _Complex float c128ibm __attribute__ ((mode (TC)));
void f_ft128_arg(long double *x);
void f_ft128_complex_arg(_Complex long double *x);
void test_TFtype(f128ibm *a) { f_ft128_arg (a); }
void test_TCtype(c128ibm *a) { f_ft128_complex_arg (a); }
typedef float w128ibm __attribute__((mode(IF)));
typedef _Complex float cw128ibm __attribute__((mode(IC)));
void f_ibm128_arg(__ibm128 *x);
void f_ibm128_complex_arg(_Complex __ibm128 *x);
void test_IFtype(w128ibm *a) { f_ibm128_arg(a); }
void test_ICtype(cw128ibm *a) { f_ibm128_complex_arg(a); }
#elif TEST_F128_PPC64
typedef int invalid_7 __attribute((mode(KF))); // expected-error{{type of machine mode does not match type of base type}}
typedef int invalid_8 __attribute((mode(KI))); // expected-error{{unknown machine mode}}
typedef _Complex float cf128 __attribute__((mode(KC)));
typedef float f128 __attribute__((mode(KF)));
void f_f128_arg(__float128 *x);
void f_f128_complex_arg(_Complex __float128 *x);
void test_KFtype(f128 *a) { f_f128_arg(a); }
void test_KCtype(cf128 *a) { f_f128_complex_arg(a); }
#elif TEST_MIPS_32
typedef unsigned int gcc_unwind_word __attribute__((mode(unwind_word)));
int foo[sizeof(gcc_unwind_word) == 4 ? 1 : -1];
#elif TEST_MIPS_N32
typedef unsigned int gcc_unwind_word __attribute__((mode(unwind_word)));
int foo[sizeof(gcc_unwind_word) == 8 ? 1 : -1];
#elif TEST_MIPS_64
typedef unsigned int gcc_unwind_word __attribute__((mode(unwind_word)));
int foo[sizeof(gcc_unwind_word) == 8 ? 1 : -1];
#else
#error Unknown test architecture.
#endif

struct S {
  int n __attribute((mode(HI)));
};
