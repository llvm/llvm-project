// RUN: %check_clang_tidy -std=c++11-or-later %s portability-avoid-platform-specific-fundamental-types %t -- -config="{CheckOptions: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnInts, value: false}, {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, value: false}]}"

// Test character types that should trigger warnings when WarnOnChars is enabled
char global_char = 'a';
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

signed char global_signed_char = 'b';
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

unsigned char global_unsigned_char = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

// Function parameters
void func_with_char_param(char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

void func_with_signed_char_param(signed char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

void func_with_unsigned_char_param(unsigned char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

// Function return types
char func_returning_char() { return 'a'; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

signed char func_returning_signed_char() { return 'b'; }
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

unsigned char func_returning_unsigned_char() { return 'c'; }
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

// Struct fields
struct TestStruct {
  char field_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]
  
  signed char field_signed_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]
  
  unsigned char field_unsigned_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]
};

// Typedefs
typedef char char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

typedef signed char signed_char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

typedef unsigned char unsigned_char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

// Type aliases (C++11)
using char_alias = char;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent character type 'char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

using signed_char_alias = signed char;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent character type 'signed char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

using unsigned_char_alias = unsigned char;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: avoid using platform-dependent character type 'unsigned char'; consider using char8_t for text or std::byte for bytes [portability-avoid-platform-specific-fundamental-types]

// Test that integer and float types are NOT flagged when their options are disabled
int should_not_warn_int = 42;
long should_not_warn_long = 100L;
float should_not_warn_float = 3.14f;
double should_not_warn_double = 2.71828;
