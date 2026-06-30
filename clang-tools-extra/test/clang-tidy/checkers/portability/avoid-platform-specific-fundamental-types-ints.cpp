// RUN: %check_clang_tidy -std=c++11-or-later %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}]}"
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=,WIN32 %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t-win32 -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}]}" \
// RUN: -- --target=i686-pc-win32
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=FAST %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t-fast -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.IntegerReplacementStyle, \
// RUN: value: Fast}]}"
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=FAST,WIN32_FAST %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t-fast-win32 -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.IntegerReplacementStyle, \
// RUN: value: Fast}]}" \
// RUN: -- --target=i686-pc-win32
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=LEAST %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t-least -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.IntegerReplacementStyle, \
// RUN: value: Least}]}"
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=LEAST,WIN32_LEAST %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t-least-win32 -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.IntegerReplacementStyle, \
// RUN: value: Least}]}" \
// RUN: -- --target=i686-pc-win32

// Mock fixed-width integer types
// NOLINTBEGIN(portability-avoid-platform-specific-fundamental-types)
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

// Mock standard library semantic types
typedef long ptrdiff_t;
// MSVC defines size_t automatically
#ifndef _MSC_VER
typedef unsigned long size_t;
#endif
// NOLINTEND(portability-avoid-platform-specific-fundamental-types)

// === Core integer types - tested across all replacement styles ===

int global_int = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int32_t global_int = 42;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: int_fast32_t global_int = 42;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: int_least32_t global_int = 42;

short global_short = 10;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int16_t global_short = 10;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: int_fast16_t global_short = 10;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_least16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: int_least16_t global_short = 10;

long global_long = 100L;
// CHECK-MESSAGES-WIN32: :[[@LINE-1]]:1: note: 'int64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t global_long = 100L;
// CHECK-MESSAGES-WIN32_FAST: :[[@LINE-4]]:1: note: 'int_fast64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
// CHECK-MESSAGES-FAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: int_fast64_t global_long = 100L;
// CHECK-MESSAGES-WIN32_LEAST: :[[@LINE-7]]:1: note: 'int_least64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
// CHECK-MESSAGES-LEAST: :[[@LINE-8]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_least64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: int_least64_t global_long = 100L;

long long global_long_long = 1000LL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t global_long_long = 1000LL;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: int_fast64_t global_long_long = 1000LL;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int_least64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: int_least64_t global_long_long = 1000LL;

unsigned int global_unsigned_int = 42U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint32_t global_unsigned_int = 42U;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: uint_fast32_t global_unsigned_int = 42U;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: uint_least32_t global_unsigned_int = 42U;

unsigned short global_unsigned_short = 10U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using 'uint16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint16_t global_unsigned_short = 10U;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using 'uint_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: uint_fast16_t global_unsigned_short = 10U;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using 'uint_least16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: uint_least16_t global_unsigned_short = 10U;

unsigned long global_unsigned_long = 100UL;
// CHECK-MESSAGES-WIN32: :[[@LINE-1]]:1: note: 'uint64_t' suggested for compatibility with Unix, which uses 64-bit 'unsigned long'
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint64_t global_unsigned_long = 100UL;
// CHECK-MESSAGES-WIN32_FAST: :[[@LINE-4]]:1: note: 'uint_fast64_t' suggested for compatibility with Unix, which uses 64-bit 'unsigned long'
// CHECK-MESSAGES-FAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: uint_fast64_t global_unsigned_long = 100UL;
// CHECK-MESSAGES-WIN32_LEAST: :[[@LINE-7]]:1: note: 'uint_least64_t' suggested for compatibility with Unix, which uses 64-bit 'unsigned long'
// CHECK-MESSAGES-LEAST: :[[@LINE-8]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint_least64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: uint_least64_t global_unsigned_long = 100UL;

unsigned long long global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint64_t global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using 'uint_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: uint_fast64_t global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using 'uint_least64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: uint_least64_t global_unsigned_long_long = 1000ULL;

// === Types that should NEVER trigger warnings ===
bool global_bool = true;

// Character and float types should NOT be flagged when their options are disabled
float should_not_warn_float = 3.14f;
double should_not_warn_double = 2.71828;
char should_not_warn_char = 'a';
signed char should_not_warn_signed_char = 'b';
unsigned char should_not_warn_unsigned_char = 'c';

// Fixed-width types should NOT trigger warnings
uint32_t global_uint32 = 42U;
int32_t global_int32 = 42;
uint64_t global_uint64 = 100ULL;
int64_t global_int64 = 100LL;

// Standard library semantic types should NOT trigger warnings
size_t global_size = 100;
ptrdiff_t global_ptrdiff = 50;

// === Function parameters - all three styles ===

void function_with_int_param(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void function_with_int_param(int32_t param) {
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:30: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: void function_with_int_param(int_fast32_t param) {
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:30: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: void function_with_int_param(int_least32_t param) {
}

void function_with_short_param(short param) {
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void function_with_short_param(int16_t param) {
}

// === Function return types - all three styles ===

int function_returning_int() {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int32_t function_returning_int() {
// CHECK-MESSAGES-FAST: :[[@LINE-3]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-FAST: int_fast32_t function_returning_int() {
// CHECK-MESSAGES-LEAST: :[[@LINE-5]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES-LEAST: int_least32_t function_returning_int() {
    return 42;
}

long function_returning_long() {
// CHECK-MESSAGES-WIN32: :[[@LINE-1]]:1: note: 'int64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t function_returning_long() {
    return 100L;
}

// === Local variables - all three styles ===

void test_local_variables() {
    int local_int = 10;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int32_t local_int = 10;
    // CHECK-MESSAGES-FAST: :[[@LINE-3]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-FAST: int_fast32_t local_int = 10;
    // CHECK-MESSAGES-LEAST: :[[@LINE-5]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-LEAST: int_least32_t local_int = 10;

    short local_short = 5;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int16_t local_short = 5;
    // CHECK-MESSAGES-FAST: :[[@LINE-3]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-FAST: int_fast16_t local_short = 5;
    // CHECK-MESSAGES-LEAST: :[[@LINE-5]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_least16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-LEAST: int_least16_t local_short = 5;

    unsigned long local_unsigned_long = 200UL;
    // CHECK-MESSAGES-WIN32: :[[@LINE-1]]:5: note: 'uint64_t' suggested for compatibility with Unix, which uses 64-bit 'unsigned long'
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: uint64_t local_unsigned_long = 200UL;

    // These should not trigger warnings
    char local_char = 'x';
    bool local_bool = false;

    // Fixed-width types should not trigger warnings
    uint32_t local_uint32 = 42U;
    int64_t local_int64 = 100LL;

    // Standard library semantic types should not trigger warnings
    size_t local_size = 200;
    ptrdiff_t local_ptrdiff = 10;
}

// === Struct/class members - all three styles ===

struct TestStruct {
    int member_int;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int32_t member_int;
    // CHECK-MESSAGES-FAST: :[[@LINE-3]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-FAST: int_fast32_t member_int;
    // CHECK-MESSAGES-LEAST: :[[@LINE-5]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_least32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-LEAST: int_least32_t member_int;

    long member_long;
    // CHECK-MESSAGES-WIN32: :[[@LINE-1]]:5: note: 'int64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int64_t member_long;
    // CHECK-MESSAGES-WIN32_FAST: :[[@LINE-4]]:5: note: 'int_fast64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
    // CHECK-MESSAGES-FAST: :[[@LINE-5]]:5: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-FAST: int_fast64_t member_long;
    // CHECK-MESSAGES-WIN32_LEAST: :[[@LINE-7]]:5: note: 'int_least64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
    // CHECK-MESSAGES-LEAST: :[[@LINE-8]]:5: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_least64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES-LEAST: int_least64_t member_long;

    // These should not trigger warnings
    char member_char;
    bool member_bool;
};

class TestClass {
public:
    unsigned int public_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: uint32_t public_member;

private:
    short private_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int16_t private_member;
};

// === Typedefs and type aliases ===

typedef int MyInt;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: typedef int32_t MyInt;

using MyLong = long;
// CHECK-MESSAGES-WIN32: :[[@LINE-1]]:16: note: 'int64_t' suggested for compatibility with Unix, which uses 64-bit 'long'
// CHECK-MESSAGES: :[[@LINE-2]]:16: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: using MyLong = int64_t;

typedef long long customType;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: typedef int64_t customType;

// === Template specializations ===

template<typename T>
void template_function(T param) {}

template<>
void template_function<int>(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-MESSAGES: :[[@LINE-2]]:29: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void template_function<int32_t>(int32_t param) {
}
