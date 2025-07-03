// RUN: %check_clang_tidy %s modernize-avoid-fundamental-integer-types %t

// Test fundamental integer types that should trigger warnings
int global_int = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

short global_short = 10;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent fundamental integer type 'short'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

long global_long = 100L;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using platform-dependent fundamental integer type 'long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

long long global_long_long = 1000LL;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent fundamental integer type 'long long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

unsigned int global_unsigned_int = 42U;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

unsigned short global_unsigned_short = 10U;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: avoid using platform-dependent fundamental integer type 'unsigned short'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

unsigned long global_unsigned_long = 100UL;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

unsigned long long global_unsigned_long_long = 1000ULL;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using platform-dependent fundamental integer type 'unsigned long long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

// Test semantic types that should NOT trigger warnings
char global_char = 'a';
signed char global_signed_char = 'b';
unsigned char global_unsigned_char = 'c';
bool global_bool = true;
wchar_t global_wchar = L'w';

// Test function parameters
void function_with_int_param(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
}

void function_with_short_param(short param) {
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: avoid using platform-dependent fundamental integer type 'short'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
}

// Test function return types
int function_returning_int() {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    return 42;
}

long function_returning_long() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using platform-dependent fundamental integer type 'long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    return 100L;
}

// Test local variables
void test_local_variables() {
    int local_int = 10;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
    short local_short = 5;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent fundamental integer type 'short'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
    unsigned long local_unsigned_long = 200UL;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: avoid using platform-dependent fundamental integer type 'unsigned long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
    // These should not trigger warnings
    char local_char = 'x';
    bool local_bool = false;
}

// Test struct/class members
struct TestStruct {
    int member_int;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
    long member_long;
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid using platform-dependent fundamental integer type 'long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
    // These should not trigger warnings
    char member_char;
    bool member_bool;
};

class TestClass {
public:
    unsigned int public_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: avoid using platform-dependent fundamental integer type 'unsigned int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
    
private:
    short private_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent fundamental integer type 'short'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
};

// Test typedefs and type aliases
typedef int MyInt;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

using MyLong = long;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent fundamental integer type 'long'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]

// Test template parameters
template<typename T>
void template_function(T param) {}

template<>
void template_function<int>(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: avoid using platform-dependent fundamental integer type 'int'; consider using a typedef or fixed-width type instead [modernize-avoid-fundamental-integer-types]
}