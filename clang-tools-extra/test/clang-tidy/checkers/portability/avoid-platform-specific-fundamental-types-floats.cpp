// RUN: %check_clang_tidy -std=c++23-or-later %s portability-avoid-platform-specific-fundamental-types %t -- -config="{CheckOptions: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnInts, value: false}, {key: portability-avoid-platform-specific-fundamental-types.WarnOnChars, value: false}]}"

// Mock fixed-width float types
// In reality, these types are aliases to "extended floating point types", and
// are not typedefs. However, there isn't a good way to mock extended floats as
// they are not fundamental types.
// NOLINTBEGIN(portability-avoid-platform-specific-fundamental-types)
typedef float float32_t;
typedef double float64_t;
// NOLINTEND(portability-avoid-platform-specific-fundamental-types)

// Test floating point types that should trigger warnings when WarnOnFloats is enabled
float global_float = 3.14f;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: float32_t global_float = 3.14f;

double global_double = 3.14159;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
// CHECK-FIXES: float64_t global_double = 3.14159;


// Test function parameters with float types
void function_with_float_param(float param) {
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
}

void function_with_double_param(double param) {
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
}

// Test function return types with float types
float function_returning_float() {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
    return 3.14f;
}

double function_returning_double() {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
    return 3.14159;
}

// Test local variables with float types
void test_local_float_variables() {
    float local_float = 2.71f;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
    
    double local_double = 2.71828;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
    
    // Fixed-width types should not trigger warnings
    float32_t local_float32 = 3.14f;
    float64_t local_float64 = 3.14159;
}

// Test struct/class members with float types
struct TestFloatStruct {
    float member_float;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
    
    double member_double;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
    
    // Fixed-width types should not trigger warnings
    float32_t member_float32;
    float64_t member_float64;
};

class TestFloatClass {
public:
    float public_float_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
    
private:
    double private_double_member;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
};

// Test typedefs and type aliases with float types
typedef float MyFloat;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]

using MyDouble = double;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]

// Test template specializations with float types
template<typename T>
void template_function(T param) {}

template<>
void template_function<float>(float param) {
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: avoid using platform-dependent floating point type 'float'; consider using 'float32_t' instead [portability-avoid-platform-specific-fundamental-types]
}

template<>
void template_function<double>(double param) {
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: avoid using platform-dependent floating point type 'double'; consider using 'float64_t' instead [portability-avoid-platform-specific-fundamental-types]
}

// Test that integer and char types are NOT flagged when their options are disabled
int should_not_warn_int = 42;
long should_not_warn_long = 100L;
char should_not_warn_char = 'a';
signed char should_not_warn_signed_char = 'b';
unsigned char should_not_warn_unsigned_char = 'c';
