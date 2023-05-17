// RUN: %check_clang_tidy %s llvm-qualified-auto %t

// This check just ensures by default the llvm alias doesn't add const
// qualifiers to decls, so no need to copy the entire test file from
// readability-qualified-auto.

int *getIntPtr();
const int *getCIntPtr();

void foo() {
  auto NakedPtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto NakedPtr' can be declared as 'auto *NakedPtr'
  // CHECK-FIXES: {{^}}  auto *NakedPtr = getIntPtr();
  auto NakedConstPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto NakedConstPtr' can be declared as 'const auto *NakedConstPtr'
  // CHECK-FIXES: {{^}}  const auto *NakedConstPtr = getCIntPtr();
  auto *Ptr = getIntPtr();
  auto *ConstPtr = getCIntPtr();
  auto &NakedRef = *getIntPtr();
  auto &NakedConstRef = *getCIntPtr();
}
