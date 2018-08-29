// RUN: %check_clang_tidy %s abseil-no-internal-dependencies %t,  -- -- -I %S/Inputs
// RUN: clang-tidy -checks='-*, abseil-no-internal-dependencies' -header-filter='.*' %s -- -I %S/Inputs 2>&1 | FileCheck %s

#include "absl/strings/internal-file.h"
// CHECK-NOT: warning:

#include "absl/external-file.h"
// CHECK: absl/external-file.h:6:24: warning: do not reference any 'internal' namespaces; those implementation details are reserved to Abseil [abseil-no-internal-dependencies]

void DirectAcess() {
  absl::strings_internal::InternalFunction();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not reference any 'internal' namespaces; those implementation details are reserved to Abseil

  absl::strings_internal::InternalTemplateFunction<std::string>("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not reference any 'internal' namespaces; those implementation details are reserved to Abseil
}

class FriendUsage {
  friend struct absl::container_internal::InternalStruct;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: do not reference any 'internal' namespaces; those implementation details are reserved to Abseil
};

namespace absl {
void OpeningNamespace() {
  strings_internal::InternalFunction();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not reference any 'internal' namespaces; those implementation details are reserved to Abseil
}
} // namespace absl

// should not trigger warnings
void CorrectUsage() {
  std::string Str = absl::StringsFunction("a");
  absl::SomeContainer b;
}

namespace absl {
SomeContainer b;
std::string Str = absl::StringsFunction("a");
} // namespace absl
