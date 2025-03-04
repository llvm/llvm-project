//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

// When the fmt is not a format string for Args the expression is not a core
// constant exprssion. In this case the code is ill-formed.
// When this happens due to issues with the formatter specialization libc++
// diagnoses the issue.

#include <format>
#include <variant>

struct no_formatter_specialization {};
void test_no_formatter_specialization() {
  no_formatter_specialization t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization has not been provided.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization has not been provided.}}
  (void)std::format(L"{}", t);
#endif
}

struct correct_formatter_specialization {};
template <class CharT>
struct std::formatter<correct_formatter_specialization, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(correct_formatter_specialization&, FormatContext& ctx) const {
    return ctx.out();
  }
};
void test_correct_formatter_specialization() {
  correct_formatter_specialization t;
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  (void)std::format(L"{}", t);
#endif
}

struct formatter_not_semiregular {};
template <class CharT>
struct std::formatter<formatter_not_semiregular, CharT> {
  formatter(int);

  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(formatter_not_semiregular&, FormatContext&) const;
};
void test_formatter_not_semiregular() {
  formatter_not_semiregular t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization is not semiregular.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization is not semiregular.}}
  (void)std::format(L"{}", t);
#endif
}

struct formatter_no_parse_function {};
template <class CharT>
struct std::formatter<formatter_no_parse_function, CharT> {
  template <class FormatContext>
  typename FormatContext::iterator format(formatter_no_parse_function&, FormatContext&) const;
};
void test_formatter_no_parse_function() {
  formatter_no_parse_function t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format(L"{}", t);
#endif
}

struct parse_function_invalid_arguments {};
template <class CharT>
struct std::formatter<parse_function_invalid_arguments, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx, int) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(parse_function_invalid_arguments&, FormatContext&) const;
};
void test_parse_function_invalid_arguments() {
  parse_function_invalid_arguments t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format(L"{}", t);
#endif
}

struct parse_function_invalid_return_type {};
template <class CharT>
struct std::formatter<parse_function_invalid_return_type, CharT> {
  template <class ParseContext>
  constexpr int parse(ParseContext&) {
    return 42;
  }

  template <class FormatContext>
  typename FormatContext::iterator format(parse_function_invalid_return_type&, FormatContext&) const;
};
void test_parse_function_invalid_return_type() {
  parse_function_invalid_return_type t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's parse function does not return the required type.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's parse function does not return the required type.}}
  (void)std::format(L"{}", t);
#endif
}

struct no_format_function {};
template <class CharT>
struct std::formatter<no_format_function, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }
};
void test_no_format_function() {
  no_format_function t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::format(L"{}", t);
#endif
}

struct format_function_invalid_arguments {};
template <class CharT>
struct std::formatter<format_function_invalid_arguments, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(format_function_invalid_arguments&) const;
};
void test_format_function_invalid_arguments() {
  format_function_invalid_arguments t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  (void)std::format(L"{}", t);
#endif
}

struct format_function_invalid_return_type {};
template <class CharT>
struct std::formatter<format_function_invalid_return_type, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  int format(format_function_invalid_return_type&, FormatContext&) const;
};
void test_format_function_invalid_return_type() {
  format_function_invalid_return_type t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function does not return the required type.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function does not return the required type.}}
  (void)std::format(L"{}", t);
#endif
}

struct format_function_not_const_qualified {};
template <class CharT>
struct std::formatter<format_function_not_const_qualified, CharT> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(format_function_not_const_qualified&, FormatContext&);
};
void test_format_function_not_const_qualified() {
  format_function_not_const_qualified t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function is not const qualified.}}
  (void)std::format("{}", t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // expected-error@*:* {{static assertion failed: The required formatter specialization's format function is not const qualified.}}
  (void)std::format(L"{}", t);
#endif
}

struct auto_deduction_correct_formatter_specialization
    : std::variant<auto_deduction_correct_formatter_specialization*> {
  auto_deduction_correct_formatter_specialization* p = nullptr;
  constexpr const std::variant<auto_deduction_correct_formatter_specialization*>& decay() const noexcept {
    return *this;
  }
};

template <>
struct std::formatter<auto_deduction_correct_formatter_specialization, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }
  static constexpr auto format(const auto_deduction_correct_formatter_specialization& x, auto& ctx) {
    if (!x.p)
      return ctx.out();
    auto m = [&](const auto_deduction_correct_formatter_specialization* t) {
      return std::format_to(ctx.out(), "{}", *t);
    };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_correct_formatter_specialization() {
  auto_deduction_correct_formatter_specialization t;
  (void)std::format("{}", t);
}

struct auto_deduction_no_parse_function : std::variant<auto_deduction_no_parse_function*> {
  auto_deduction_no_parse_function* p = nullptr;
  constexpr const std::variant<auto_deduction_no_parse_function*>& decay() const noexcept { return *this; }
};

template <>
struct std::formatter<auto_deduction_no_parse_function, char> {
  static constexpr auto format(const auto_deduction_no_parse_function& x, auto& ctx) {
    if (!x.p)
      return ctx.out();
    auto m = [&](const auto_deduction_no_parse_function* t) { return std::format_to(ctx.out(), "{}", *t); };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_no_parse_function() {
  auto_deduction_no_parse_function t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format("{}", t);
}

struct auto_deduction_parse_function_invalid_arguments
    : std::variant<auto_deduction_parse_function_invalid_arguments*> {
  auto_deduction_parse_function_invalid_arguments* p = nullptr;
  constexpr const std::variant<auto_deduction_parse_function_invalid_arguments*>& decay() const noexcept {
    return *this;
  }
};

template <>
struct std::formatter<auto_deduction_parse_function_invalid_arguments, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext& ctx, int) {
    return ctx.begin();
  }
  static constexpr auto format(const auto_deduction_parse_function_invalid_arguments& x, auto& ctx) {
    if (!x.p)
      return ctx.out();
    auto m = [&](const auto_deduction_parse_function_invalid_arguments* t) {
      return std::format_to(ctx.out(), "{}", *t);
    };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_parse_function_invalid_arguments() {
  auto_deduction_parse_function_invalid_arguments t;
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a parse function taking the proper arguments.}}
  (void)std::format("{}", t);
}

struct auto_deduction_parse_function_invalid_return_types
    : std::variant<auto_deduction_parse_function_invalid_return_types*> {
  auto_deduction_parse_function_invalid_return_types* p = nullptr;
  constexpr const std::variant<auto_deduction_parse_function_invalid_return_types*>& decay() const noexcept {
    return *this;
  }
};

template <>
struct std::formatter<auto_deduction_parse_function_invalid_return_types, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext&) {
    return 42;
  }
  static constexpr auto format(const auto_deduction_parse_function_invalid_return_types& x, auto& ctx) {
    if (!x.p)
      return ctx.out();
    auto m = [&](const auto_deduction_parse_function_invalid_return_types* t) {
      return std::format_to(ctx.out(), "{}", *t);
    };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_parse_function_invalid_return_types() {
  // expected-error@*:* {{static assertion failed: The required formatter specialization's parse function does not return the required type.}}
  auto_deduction_parse_function_invalid_return_types t;
  (void)std::format("{}", t);
}

struct auto_deduction_no_format_function : std::variant<auto_deduction_no_format_function*> {
  auto_deduction_no_format_function* p = nullptr;
  // expected-error@*:* 2 {{static assertion failed: The required formatter specialization's format function does not return the required type.}}
  constexpr const std::variant<auto_deduction_no_format_function*>& decay() const noexcept { return *this; }
};

template <>
struct std::formatter<auto_deduction_no_format_function, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }
};

void test_auto_deduction_no_format_function() {
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  auto_deduction_no_format_function t;
  (void)std::format("{}", t);
}

struct auto_deduction_format_function_invalid_arguments
    : std::variant<auto_deduction_format_function_invalid_arguments*> {
  auto_deduction_format_function_invalid_arguments* p = nullptr;
  constexpr const std::variant<auto_deduction_format_function_invalid_arguments*>& decay() const noexcept {
    return *this;
  }
};

template <>
struct std::formatter<auto_deduction_format_function_invalid_arguments, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }
  static constexpr auto format(const auto_deduction_format_function_invalid_arguments& x, auto& ctx, int) {
    if (!x.p)
      return ctx.out();
    auto m = [&](const auto_deduction_format_function_invalid_arguments* t) {
      return std::format_to(ctx.out(), "{}", *t);
    };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_format_function_invalid_arguments() {
  // expected-error@*:* {{static assertion failed: The required formatter specialization does not have a format function taking the proper arguments.}}
  auto_deduction_format_function_invalid_arguments t;
  (void)std::format("{}", t);
}

struct auto_deduction_format_function_invalid_return_types
    : std::variant<auto_deduction_format_function_invalid_return_types*> {
  auto_deduction_format_function_invalid_return_types* p = nullptr;
  constexpr const std::variant<auto_deduction_format_function_invalid_return_types*>& decay() const noexcept {
    return *this;
  }
};

template <>
struct std::formatter<auto_deduction_format_function_invalid_return_types, char> {
  template <class ParseContext>
  static constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }
  static constexpr auto format(const auto_deduction_format_function_invalid_return_types& x, auto& ctx) {
    if (!x.p)
      return 42;
    auto m = [&](const auto_deduction_format_function_invalid_return_types* t) {
      std::format_to(ctx.out(), "{}", *t);
      return 42;
    };
    return std::visit(m, x.decay());
  }
};

void test_auto_deduction_format_function_invalid_return_types() {
  auto_deduction_format_function_invalid_return_types t;
  (void)std::format("{}", t);
}
