// RUN: %check_clang_tidy -std=c++11-or-later %s performance-expensive-value-or %t \
// RUN:   -config='{CheckOptions: {performance-expensive-value-or.WarnOnRvalueOptional: true}}'

#include <optional>
#include <string>

std::optional<std::string> getOpt();
void positiveRvalueOptional() {
  auto val = getOpt().value_or("default");
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'value_or' copies expensive type 'std::basic_string<char>'; consider using 'operator*' or 'value()' with a separate fallback [performance-expensive-value-or]
}
