// RUN: %check_clang_tidy -std=c++20-or-later %s \
// RUN: portability-avoid-platform-specific-fundamental-types %t -- \
// RUN: -config="{CheckOptions: \
// RUN: [{key: portability-avoid-platform-specific-fundamental-types.WarnOnInts, \
// RUN: value: false}, \
// RUN: {key: portability-avoid-platform-specific-fundamental-types.WarnOnFloats, \
// RUN: value: false}]}"

// Mock byte
// NOLINTBEGIN(portability-avoid-platform-specific-fundamental-types)
namespace std {
enum class byte : unsigned char {};
}
// NOLINTEND(portability-avoid-platform-specific-fundamental-types)

// Test character types that should trigger warnings when WarnOnChars is enabled
char global_char = 'a';
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

signed char global_signed_char = 'b';
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

unsigned char global_unsigned_char = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Function parameters
void func_with_char_param(char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

void func_with_signed_char_param(signed char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

void func_with_unsigned_char_param(unsigned char param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Function return types
char func_returning_char() { return 'a'; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

signed char func_returning_signed_char() { return 'b'; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

unsigned char func_returning_unsigned_char() { return 'c'; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Struct fields
struct TestStruct {
  char field_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  signed char field_signed_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  unsigned char field_unsigned_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
};

// Typedefs
typedef char char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

typedef signed char signed_char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

typedef unsigned char unsigned_char_typedef;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Type aliases (C++11)
using char_alias = char;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

using signed_char_alias = signed char;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

using unsigned_char_alias = unsigned char;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Test const and reference parameters
void func_const_char_ref_param(const char &param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

void func_const_signed_char_ref_param(const signed char &param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

void func_const_unsigned_char_ref_param(const unsigned char &param) {}
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

// Test template declarations
template <typename T> struct TemplateStruct {
  typedef char char_type;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  char template_field;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  char create_char() { return char('x'); }
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  // CHECK-MESSAGES: :[[@LINE-2]]:31: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
};

// Test namespace usage
namespace ns_chars {
  char ns_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
}

void test_comprehensive_cases() {
  // Test with spacing
  char  spaced_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  signed   char   spaced_signed_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'signed char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test pointers
  char *char_ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  unsigned char *unsigned_char_ptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test static declarations
  static char static_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test cv-qualifiers
  const char const_char = 'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  volatile char volatile_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  const volatile char const_volatile_char = '\0';
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test auto with initializer
  auto auto_char = char{'x'};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Verify auto without initializer isn't flagged
  auto auto_nonexplicit_char = 'x';
  
  char brace_init_char{};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test arrays
  char char_array[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
  
  unsigned char unsigned_char_array[] = {'a', 'b', 'c'};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'unsigned char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

  // Test multiple declarations
  char multi_char1, multi_char2, multi_char3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
}

// Test class with character type members
class CharacterClass {
public:
  char &get_char_ref();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]

private:
  char private_char;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
};

// Test template instantiation
void test_template_instantiation() {
  TemplateStruct<int> template_instance;
  char template_test = template_instance.create_char();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes [portability-avoid-platform-specific-fundamental-types]
}

// Test template specializations with char types
template<typename T>
void template_function(T param) {}

template<>
void template_function<char>(char param) {
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes 
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: avoid using platform-dependent character type 'char'; consider using 'char8_t' for text or 'std::byte' for bytes 
}

// Test that integer and float types are NOT flagged when their options are disabled
int should_not_warn_int = 42;
long should_not_warn_long = 100L;
float should_not_warn_float = 3.14f;
double should_not_warn_double = 2.71828;

// Test that unicode characters are not flagged
char8_t utf8 = u8'a';
char16_t utf16 = u'a';
char32_t utf32 = U'a';
std::byte null{0};
