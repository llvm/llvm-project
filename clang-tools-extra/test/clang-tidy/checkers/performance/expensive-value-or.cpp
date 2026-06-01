// RUN: %check_clang_tidy -std=c++11-or-later %s performance-expensive-value-or %t \
// RUN:   -config='{CheckOptions: {performance-expensive-value-or.OptionalTypes: "::std::optional;::absl::optional;::custom::CamelOptional;::custom::PascalOptional"}}'

#include <optional>
#include <string>
#include <utility>

void positiveNonTriviallyCopyable(std::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

struct LargeStruct {
  char data[128];
};

void positiveLargeStruct(std::optional<LargeStruct> opt) {
  auto val = opt.value_or(LargeStruct{});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'LargeStruct'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

struct SmallNonTrivial {
  int x;
  SmallNonTrivial(int x) : x(x) {}
  SmallNonTrivial(const SmallNonTrivial &o) : x(o.x) {}
};

void positiveSmallNonTrivial(std::optional<SmallNonTrivial> opt) {
  auto val = opt.value_or(SmallNonTrivial{42});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'SmallNonTrivial'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

void consume(std::string s);
void positiveDirectUse(std::optional<std::string> opt) {
  consume(opt.value_or("fallback"));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

template <typename T> void positiveTemplate(std::optional<T> opt) {
  auto val = opt.value_or(T{});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}
void instantiate() {
  positiveTemplate(std::optional<std::string>{});
}

using OptString = std::optional<std::string>;
void positiveTypeAlias(OptString opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

void positiveNested(std::optional<std::optional<std::string>> opt) {
  auto val = opt.value_or(std::optional<std::string>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::optional<std::basic_string<char>>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

void positiveMoveResult(std::optional<std::string> opt) {
  auto val = std::move(opt.value_or("default"));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

void negativeInt(std::optional<int> opt) {
  auto val = opt.value_or(0);
}

struct SmallPOD {
  char x;
  char y;
};

void negativeSmallPOD(std::optional<SmallPOD> opt) {
  auto val = opt.value_or(SmallPOD{0, 0});
}

struct SixteenBytes {
  char d[16];
};

void negativeBoundary(std::optional<SixteenBytes> opt) {
  auto val = opt.value_or(SixteenBytes{});
}

struct TrivialCopyNonTrivialAssign {
  int x;
  TrivialCopyNonTrivialAssign() = default;
  TrivialCopyNonTrivialAssign(const TrivialCopyNonTrivialAssign &) = default;
  TrivialCopyNonTrivialAssign &operator=(const TrivialCopyNonTrivialAssign &);
  ~TrivialCopyNonTrivialAssign() = default;
};

void negativeTrivialCopyNonTrivialAssign(
    std::optional<TrivialCopyNonTrivialAssign> opt) {
  auto val = opt.value_or(TrivialCopyNonTrivialAssign{});
}

std::optional<std::string> getOpt();
void negativeRvalueOptional() {
  auto val = getOpt().value_or("default");
}

std::optional<LargeStruct> getLargeOpt();
void positiveRvalueNoMoveAdvantage() {
  auto val = getLargeOpt().value_or(LargeStruct{});
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: 'value_or' copies expensive type 'LargeStruct'
}

namespace absl {
template <typename T> class optional {
public:
  T value_or(T default_value) const;
};
} // namespace absl

void positiveAbslDefault(absl::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

namespace custom {
template <typename T> class CamelOptional {
public:
  T valueOr(T default_value) const;
};
template <typename T> class PascalOptional {
public:
  T ValueOr(T default_value) const;
};
} // namespace custom

void positiveValueOr(custom::CamelOptional<std::string> opt) {
  auto val = opt.valueOr("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'valueOr' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}

void positiveValueOrPascal(custom::PascalOptional<std::string> opt) {
  auto val = opt.ValueOr("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'ValueOr' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}
