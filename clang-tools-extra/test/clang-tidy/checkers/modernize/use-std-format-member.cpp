// RUN: %check_clang_tidy %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             { \
// RUN:               modernize-use-std-format.StrFormatLikeFunctions: 'MyClass::StrFormat', \
// RUN:               modernize-use-std-format.ReplacementFormatFunction: 'format', \
// RUN:             } \
// RUN:            }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cstdio>
#include <string.h>
#include <string>

struct MyClass
{
  template <typename S, typename... Args>
  std::string StrFormat(const S &format, const Args&... args);
};

std::string StrFormat_simple(MyClass &myclass, MyClass *pmyclass) {
  std::string s;

  s += myclass.StrFormat("MyClass::StrFormat dot %d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: s += myclass.format("MyClass::StrFormat dot {}", 42);

  s += pmyclass->StrFormat("MyClass::StrFormat pointer %d", 43);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: s += pmyclass->format("MyClass::StrFormat pointer {}", 43);

  s += (*pmyclass).StrFormat("MyClass::StrFormat deref pointer %d", 44);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: s += (*pmyclass).format("MyClass::StrFormat deref pointer {}", 44);

  return s;
}

struct MyDerivedClass : public MyClass {};

std::string StrFormat_derived(MyDerivedClass &derived) {
  return derived.StrFormat("MyDerivedClass::StrFormat dot %d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'format' instead of 'StrFormat' [modernize-use-std-format]
  // CHECK-FIXES: return derived.format("MyDerivedClass::StrFormat dot {}", 42);
}
