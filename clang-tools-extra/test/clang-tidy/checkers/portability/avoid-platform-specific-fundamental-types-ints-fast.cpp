// RUN: %check_clang_tidy -std=c++11-or-later %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.IntegerReplacementStyle, \
// RUN: value: Fast}]}"

// Test "Fast" replacement style
int global_int = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int_fast32_t global_int = 42;

unsigned int global_unsigned_int = 42U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint_fast32_t global_unsigned_int = 42U;

short global_short = 10;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int_fast16_t global_short = 10;

unsigned short global_unsigned_short = 10U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using 'uint_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint_fast16_t global_unsigned_short = 10U;

long global_long = 100L;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int_fast64_t global_long = 100L;

unsigned long global_unsigned_long = 100UL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint_fast64_t global_unsigned_long = 100UL;

long long global_long_long = 1000LL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int_fast64_t global_long_long = 1000LL;

unsigned long long global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using 'uint_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint_fast64_t global_unsigned_long_long = 1000ULL;

// Test function parameters
void function_with_int_param(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void function_with_int_param(int_fast32_t param) {
}

// Test function return types
int function_returning_int() {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int_fast32_t function_returning_int() {
    return 42;
}

// Test local variables
void test_local_variables() {
    int local_int = 10;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int_fast32_t local_int = 10;
    
    short local_short = 5;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int_fast16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int_fast16_t local_short = 5;
}

// Test struct members
struct TestStruct {
    int member_int;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int_fast32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int_fast32_t member_int;
    
    long member_long;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int_fast64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int_fast64_t member_long;
};
