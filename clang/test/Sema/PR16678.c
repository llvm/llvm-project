// RUN: %clang_cc1 -DX32TYPE=ULONG -triple powerpc-unknown-linux-gnu -std=c89 -x c %s -verify
// RUN: %clang_cc1 -DX32TYPE=ULONG -triple powerpc-unknown-linux-gnu -std=iso9899:199409 -x c %s -verify
// RUN: %clang_cc1 -DX32TYPE=ULONG -triple powerpc-unknown-linux-gnu -std=c++98 -x c++ %s -verify
// RUN: %clang_cc1 -DX32TYPE=LLONG -triple powerpc-unknown-linux-gnu -std=c99 -x c %s -verify
// RUN: %clang_cc1 -DX32TYPE=LLONG -triple powerpc-unknown-linux-gnu -std=c11 -x c %s -verify
// RUN: %clang_cc1 -DX32TYPE=LLONG -triple powerpc-unknown-linux-gnu -std=c++11 -x c++ %s -verify
// RUN: %clang_cc1 -DX32TYPE=LLONG -triple powerpc-unknown-linux-gnu -std=c++1y -x c++ %s -verify
// RUN: %clang_cc1 -DX32TYPE=LLONG -triple powerpc-unknown-linux-gnu -std=c++1z -x c++ %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULONG -triple powerpc64-unknown-linux-gnu -std=c89 -x c %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULONG -triple powerpc64-unknown-linux-gnu -std=iso9899:199409 -x c %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULONG -triple powerpc64-unknown-linux-gnu -std=c++98 -x c++ %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULLONG -triple powerpc64-unknown-linux-gnu -std=c99 -x c %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULLONG -triple powerpc64-unknown-linux-gnu -std=c11 -x c %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULLONG -triple powerpc64-unknown-linux-gnu -std=c++11 -x c++ %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULLONG -triple powerpc64-unknown-linux-gnu -std=c++1y -x c++ %s -verify
// RUN: %clang_cc1 -DX64TYPE=ULLONG -triple powerpc64-unknown-linux-gnu -std=c++1z -x c++ %s -verify

#ifdef X64TYPE
#define X32TYPE long
#endif

#define IS_ULONG_ULONG 1
#define IS_ULONG2(X) IS_ULONG_##X
#define IS_ULONG(X) IS_ULONG2(X)

#if !defined(X64TYPE) && !IS_ULONG(X32TYPE)
// expected-no-diagnostics
#endif

typedef unsigned long ULONG;
typedef long long LLONG;
typedef unsigned long long ULLONG;


/******************************************************************************
 * Test 2^31 as a decimal literal with no suffix and with the "l" and "L" cases.
 ******************************************************************************/
extern X32TYPE x32;
extern __typeof__(2147483648) x32;
extern __typeof__(2147483648l) x32;
extern __typeof__(2147483648L) x32;

#if IS_ULONG(X32TYPE)
#if !__cplusplus

/******************************************************************************
 * Under pre-C99 ISO C, unsigned long is attempted for decimal integer literals
 * that do not have a suffix containing "u" or "U" if the literal does not fit
 * within the range of int or long. See 6.1.3.2 paragraph 5.
 ******************************************************************************/
// expected-warning@39 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will have type 'long long' in C99 onwards}}
// expected-warning@40 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will have type 'long long' in C99 onwards}}
// expected-warning@41 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will have type 'long long' in C99 onwards}}
#else

/******************************************************************************
 * Under pre-C++11 ISO C++, the same holds if the literal contains an "l" or "L"
 * in its suffix; otherwise, the behavior is undefined. See 2.13.1 [lex.icon]
 * paragraph 2.
 ******************************************************************************/
// expected-warning@39 {{integer literal is too large to be represented in type 'long' and is subject to undefined behavior under C++98, interpreting as 'unsigned long'; this literal will have type 'long long' in C++11 onwards}}
// expected-warning@40 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C++98; this literal will have type 'long long' in C++11 onwards}}
// expected-warning@41 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C++98; this literal will have type 'long long' in C++11 onwards}}
#endif
#endif


#ifdef X64TYPE

/******************************************************************************
 * Test 2^63 as a decimal literal with no suffix and with the "l" and "L" cases.
 ******************************************************************************/
extern X64TYPE x64;
extern __typeof__(9223372036854775808) x64;
extern __typeof__(9223372036854775808l) x64;
extern __typeof__(9223372036854775808L) x64;

#if IS_ULONG(X64TYPE)

#if !__cplusplus

/******************************************************************************
 * Under pre-C99 ISO C, unsigned long is attempted for decimal integer literals
 * that do not have a suffix containing "u" or "U" if the literal does not fit
 * within the range of int or long. See 6.1.3.2 paragraph 5.
 ******************************************************************************/
// expected-warning@74 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will be ill-formed in C99 onwards}}
// expected-warning@75 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will be ill-formed in C99 onwards}}
// expected-warning@76 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C89; this literal will be ill-formed in C99 onwards}}
#else

/******************************************************************************
 * Under pre-C++11 ISO C++, the same holds if the literal contains an "l" or "L"
 * in its suffix; otherwise, the behavior is undefined. See 2.13.1 [lex.icon]
 * paragraph 2.
 ******************************************************************************/
// expected-warning@74 {{integer literal is too large to be represented in type 'long' and is subject to undefined behavior under C++98, interpreting as 'unsigned long'; this literal will be ill-formed in C++11 onwards}}
// expected-warning@75 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C++98; this literal will be ill-formed in C++11 onwards}}
// expected-warning@76 {{integer literal is too large to be represented in type 'long', interpreting as 'unsigned long' per C++98; this literal will be ill-formed in C++11 onwards}}
#endif
#else

/******************************************************************************
 * The status quo in C99/C++11-and-later modes for the literals in question is
 * to interpret them as unsigned as an extension.
 ******************************************************************************/
// expected-warning@74 {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
// expected-warning@75 {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
// expected-warning@76 {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
#endif
#endif


/******************************************************************************
 * Test preprocessor arithmetic with 2^31 as a decimal literal with no suffix
 * and with the "l" and "L" cases.
 ******************************************************************************/
#if !IS_ULONG(X32TYPE)

/******************************************************************************
 * If the literal is signed without need for the modified range of the signed
 * integer types within the controlling constant expression for conditional
 * inclusion, then it will also be signed with said modified range.
 ******************************************************************************/
#define EXPR(X) ((X - X) - 1 < 0)
#else

/******************************************************************************
 * Strictly speaking, in pre-C99/C++11 ISO C/C++, the preprocessor arithmetic is
 * evaluated with the range of long/unsigned long; however, both Clang and GCC
 * evaluate using 64-bits even when long/unsigned long are 32-bits outside of
 * preprocessing.
 *
 * If the range used becomes 32-bits, then this test will enforce the treatment
 * as unsigned of the literals in question.
 *
 * Note:
 * Under pre-C99/C++11 ISO C/C++, whether the interpretation of the literal is
 * affected by the modified range of the signed and unsigned integer types
 * within the controlling constant expression for conditional inclusion is
 * unclear.
 ******************************************************************************/
#define PP_LONG_MAX ((0ul - 1ul) >> 1)
#define EXPR(X)                                                                \
  (PP_LONG_MAX >= 0x80000000 || (X - X) - 1 > 0) // either 2^31 fits into a
                                                 // preprocessor "long" or the
                                                 // literals in question are
                                                 // unsigned
#endif

#if !(EXPR(2147483648) && EXPR(2147483648l) && EXPR(2147483648L))
#error Unexpected signedness or conversion behavior
#endif
