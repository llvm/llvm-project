// RUN: %check_clang_tidy \
// RUN:   -std=c++20 %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: {StrictMode: true}}" \
// RUN:   -- -isystem %clang_tidy_headers
// RUN: %check_clang_tidy \
// RUN:   -std=c++20 %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: {StrictMode: false}}" \
// RUN:   -- -isystem %clang_tidy_headers
#include <string>
// CHECK-FIXES: #include <format>

namespace absl
{
// Use const char * for the format since the real type is hard to mock up.
template <typename... Args>
std::string StrFormat(const char *format, const Args&... args);
} // namespace absl

template <typename T>
struct iterator {
  T *operator->();
  T &operator*();
};

std::string StrFormat_simple() {
  return absl::StrFormat("Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("Hello");
}

std::string StrFormat_complex(const char *name, double value) {
  return absl::StrFormat("'%s'='%f'", name, value);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("'{}'='{:f}'", name, value);
}

std::string StrFormat_integer_conversions() {
  return absl::StrFormat("int:%d int:%d char:%c char:%c", 65, 'A', 66, 'B');
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("int:{} int:{:d} char:{:c} char:{}", 65, 'A', 66, 'B');
}

// FormatConverter is capable of removing newlines from the end of the format
// string. Ensure that isn't incorrectly happening for std::format.
std::string StrFormat_no_newline_removal() {
  return absl::StrFormat("a line\n");
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("a line\n");
}

// FormatConverter is capable of removing newlines from the end of the format
// string. Ensure that isn't incorrectly happening for std::format.
std::string StrFormat_cstr_removal(const std::string &s1, const std::string *s2) {
  return absl::StrFormat("%s %s %s %s", s1.c_str(), s1.data(), s2->c_str(), s2->data());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("{} {} {} {}", s1, s1, *s2, *s2);
}

std::string StrFormat_strict_conversion() {
  const unsigned char uc = 'A';
  return absl::StrFormat("Integer %hhd from unsigned char\n", uc);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return std::format("Integer {} from unsigned char\n", uc);
}

std::string StrFormat_field_width_and_precision() {
  auto s1 = absl::StrFormat("width only:%*d width and precision:%*.*f precision only:%.*f", 3, 42, 4, 2, 3.14159265358979323846, 5, 2.718);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("width only:{:{}} width and precision:{:{}.{}f} precision only:{:.{}f}", 42, 3, 3.14159265358979323846, 4, 2, 2.718, 5);

  auto s2 = absl::StrFormat("width and precision positional:%1$*2$.*3$f after", 3.14159265358979323846, 4, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("width and precision positional:{0:{1}.{2}f} after", 3.14159265358979323846, 4, 2);

  const int width = 10, precision = 3;
  const unsigned int ui1 = 42, ui2 = 43, ui3 = 44;
  auto s3 = absl::StrFormat("casts width only:%*d width and precision:%*.*d precision only:%.*d\n", 3, ui1, 4, 2, ui2, 5, ui3);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES-NOTSTRICT: std::format("casts width only:{:{}} width and precision:{:{}.{}} precision only:{:.{}}", ui1, 3, ui2, 4, 2, ui3, 5);
  // CHECK-FIXES-STRICT: std::format("casts width only:{:{}} width and precision:{:{}.{}} precision only:{:.{}}", static_cast<int>(ui1), 3, static_cast<int>(ui2), 4, 2, static_cast<int>(ui3), 5);

  auto s4 = absl::StrFormat("c_str removal width only:%*s width and precision:%*.*s precision only:%.*s", 3, s1.c_str(), 4, 2, s2.c_str(), 5, s3.c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("c_str removal width only:{:>{}} width and precision:{:>{}.{}} precision only:{:.{}}", s1, 3, s2, 4, 2, s3, 5);

  const std::string *ps1 = &s1, *ps2 = &s2, *ps3 = &s3;
  auto s5 = absl::StrFormat("c_str() removal pointer width only:%-*s width and precision:%-*.*s precision only:%-.*s", 3, ps1->c_str(), 4, 2, ps2->c_str(), 5, ps3->c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("c_str() removal pointer width only:{:{}} width and precision:{:{}.{}} precision only:{:.{}}", *ps1, 3, *ps2, 4, 2, *ps3, 5);

  iterator<std::string> is1, is2, is3;
  auto s6 = absl::StrFormat("c_str() removal iterator width only:%-*s width and precision:%-*.*s precision only:%-.*s", 3, is1->c_str(), 4, 2, is2->c_str(), 5, is3->c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("c_str() removal iterator width only:{:{}} width and precision:{:{}.{}} precision only:{:.{}}", *is1, 3, *is2, 4, 2, *is3, 5);

  return s1 + s2 + s3 + s4 + s5 + s6;
}

std::string StrFormat_macros() {
  // The function call is replaced even though it comes from a macro.
#define FORMAT absl::StrFormat
  auto s1 = FORMAT("Hello %d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Hello {}", 42);

  // The format string is replaced even though it comes from a macro, this
  // behaviour is required so that that <inttypes.h> macros are replaced.
#define FORMAT_STRING "Hello %s"
  auto s2 = absl::StrFormat(FORMAT_STRING, 42);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Hello {}", 42);

  // Arguments that are macros aren't replaced with their value, even if they are rearranged.
#define VALUE 3.14159265358979323846
#define WIDTH 10
#define PRECISION 4
  auto s3 = absl::StrFormat("Hello %*.*f", WIDTH, PRECISION, VALUE);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Hello {:{}.{}f}", VALUE, WIDTH, PRECISION);
}
