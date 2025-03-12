// RUN: %check_clang_tidy \
// RUN:   -std=c++20 %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-std-format.StrictMode: true}}" \
// RUN:   -- -isystem %clang_tidy_headers \
// RUN:      -DPRI_CMDLINE_MACRO="\"s\"" \
// RUN:      -D__PRI_CMDLINE_MACRO="\"s\""
// RUN: %check_clang_tidy \
// RUN:   -std=c++20 %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-std-format.StrictMode: false}}" \
// RUN:   -- -isystem %clang_tidy_headers \
// RUN:      -DPRI_CMDLINE_MACRO="\"s\"" \
// RUN:      -D__PRI_CMDLINE_MACRO="\"s\""
#include <string>
// CHECK-FIXES: #include <format>
#include <inttypes.h>

namespace absl
{
template <typename S, typename... Args>
std::string StrFormat(const S &format, const Args&... args);
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

  // Arguments that are macros aren't replaced with their value, even if they are rearranged.
#define VALUE 3.14159265358979323846
#define WIDTH 10
#define PRECISION 4
  auto s3 = absl::StrFormat("Hello %*.*f", WIDTH, PRECISION, VALUE);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Hello {:{}.{}f}", VALUE, WIDTH, PRECISION);

  const uint64_t u64 = 42;
  const uint32_t u32 = 32;
  std::string s;

  auto s4 = absl::StrFormat("Replaceable macro at end %" PRIu64, u64);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Replaceable macro at end {}", u64);

  auto s5 = absl::StrFormat("Replaceable macros in middle %" PRIu64 " %" PRIu32 "\n", u64, u32);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: std::format("Replaceable macros in middle {} {}\n", u64, u32);

// These need PRI and __PRI prefixes so that the check get as far as looking for
// where the macro comes from.
#define PRI_FMT_MACRO "s"
#define __PRI_FMT_MACRO "s"

  auto s6 = absl::StrFormat("Unreplaceable macro at end %" PRI_FMT_MACRO, s.c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro 'PRI_FMT_MACRO' [modernize-use-std-format]

  auto s7 = absl::StrFormat(__PRI_FMT_MACRO " Unreplaceable macro at beginning %s", s);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro '__PRI_FMT_MACRO' [modernize-use-std-format]

  auto s8 = absl::StrFormat("Unreplacemable macro %" PRI_FMT_MACRO " in the middle", s);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro 'PRI_FMT_MACRO' [modernize-use-std-format]

  auto s9 = absl::StrFormat("First macro is replaceable %" PRIu64 " but second one is not %" __PRI_FMT_MACRO, u64, s);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro '__PRI_FMT_MACRO' [modernize-use-std-format]

  // Needs a PRI prefix so that we get as far as looking for where the macro comes from
  auto s10 = absl::StrFormat(" macro from command line %" PRI_CMDLINE_MACRO, s);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro 'PRI_CMDLINE_MACRO' [modernize-use-std-format]

  // Needs a __PRI prefix so that we get as far as looking for where the macro comes from
  auto s11 = absl::StrFormat(" macro from command line %" __PRI_CMDLINE_MACRO, s);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro '__PRI_CMDLINE_MACRO' [modernize-use-std-format]

  // We ought to be able to fix this since the macro surrounds the whole call
  // and therefore can't change the format string independently. This is
  // required to be able to fix calls inside Catch2 macros for example.
#define SURROUND_ALL(x) x
  auto s12 = SURROUND_ALL(absl::StrFormat("Macro surrounding entire invocation %" PRIu64, u64));
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: auto s12 = SURROUND_ALL(std::format("Macro surrounding entire invocation {}", u64));

  // But having that surrounding macro shouldn't stop us ignoring an
  // unreplaceable macro elsewhere.
  auto s13 = SURROUND_ALL(absl::StrFormat("Macro surrounding entire invocation with unreplaceable macro %" PRI_FMT_MACRO, s));
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro 'PRI_FMT_MACRO' [modernize-use-std-format]

  // At the moment at least the check will replace occurrences where the
  // function name is the result of expanding a macro.
#define SURROUND_FUNCTION_NAME(x) absl:: x
  auto s14 = SURROUND_FUNCTION_NAME(StrFormat)("Hello %d", 4442);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: use 'std::format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: auto s14 = std::format("Hello {}", 4442);

  // We can't safely fix occurrences where the macro may affect the format
  // string differently in different builds.
#define SURROUND_FORMAT(x) "!" x
  auto s15 = absl::StrFormat(SURROUND_FORMAT("Hello %d"), 4443);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: unable to use 'std::format' instead of 'StrFormat' because format string contains unreplaceable macro 'SURROUND_FORMAT' [modernize-use-std-format]
}
