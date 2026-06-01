// RUN: %check_clang_tidy -std=c++11-or-later %s performance-expensive-value-or %t

#include <string>

namespace absl {
template <typename T> class optional {
public:
  T value_or(T default_value) const;
};
} // namespace absl

void positiveAbsl(absl::optional<std::string> opt) {
  auto val = opt.value_or("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}
