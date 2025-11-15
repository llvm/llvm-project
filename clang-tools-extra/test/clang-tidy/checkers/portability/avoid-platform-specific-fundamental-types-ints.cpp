// RUN: %check_clang_tidy -std=c++11-or-later %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}]}"

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

// Test fundamental integer types that should trigger warnings
int global_int = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int32_t global_int = 42;

short global_short = 10;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int16_t global_short = 10;

long global_long = 100L;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t global_long = 100L;

long long global_long_long = 1000LL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t global_long_long = 1000LL;

unsigned int global_unsigned_int = 42U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using 'uint32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint32_t global_unsigned_int = 42U;

unsigned short global_unsigned_short = 10U;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using 'uint16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint16_t global_unsigned_short = 10U;

unsigned long global_unsigned_long = 100UL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint64_t global_unsigned_long = 100UL;

unsigned long long global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: uint64_t global_unsigned_long_long = 1000ULL;

// Test integer  that should NEVER trigger warnings
bool global_bool = true;

// Test that char and float types are NOT flagged when their options are disabled
float should_not_warn_float = 3.14f;
double should_not_warn_double = 2.71828;
char should_not_warn_char = 'a';
signed char should_not_warn_signed_char = 'b';
unsigned char should_not_warn_unsigned_char = 'c';

// Test fixed-width types that should NOT trigger warnings
uint32_t global_uint32 = 42U;
int32_t global_int32 = 42;
uint64_t global_uint64 = 100ULL;
int64_t global_int64 = 100LL;

// Test semantic standard library types that should NOT trigger warnings
size_t global_size = 100;
ptrdiff_t global_ptrdiff = 50;

// Test function parameters
void function_with_int_param(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void function_with_int_param(int32_t param) {
}

void function_with_short_param(short param) {
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void function_with_short_param(int16_t param) {
}

// Test function return types
int function_returning_int() {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int32_t function_returning_int() {
    return 42;
}

long function_returning_long() {
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: int64_t function_returning_long() {
    return 100L;
}

// Test local variables
void test_local_variables() {
    int local_int = 10;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int32_t local_int = 10;
    
    short local_short = 5;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'short'; consider using 'int16_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int16_t local_short = 5;
    
    unsigned long local_unsigned_long = 200UL;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using 'uint64_t' instead [portability-avoid-platform-specific-fundamental-types]
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

// Test struct/class members
struct TestStruct {
    int member_int;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int32_t member_int;
    
    long member_long;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
    // CHECK-FIXES: int64_t member_long;
    
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

// Test typedefs and type aliases
typedef int MyInt;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: typedef int32_t MyInt;

using MyLong = long;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: avoid using platform-dependent fundamental integer type 'long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: using MyLong = int64_t;

typedef long long customType;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using 'int64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: typedef int64_t customType;

// Test template parameters
template<typename T>
void template_function(T param) {}

template<>
void template_function<int>(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-MESSAGES: :[[@LINE-2]]:29: warning: avoid using platform-dependent fundamental integer type 'int'; consider using 'int32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: void template_function<int32_t>(int32_t param) {
}
