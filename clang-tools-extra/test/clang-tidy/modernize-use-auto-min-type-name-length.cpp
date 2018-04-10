// RUN: %check_clang_tidy %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-auto.MinTypeNameLength, value: '5'}]}" \
// RUN:   -- -std=c++11 -frtti

extern int foo();

using VeryVeryVeryLongTypeName = int;

int bar() {
  int a = static_cast<VeryVeryVeryLongTypeName>(foo());
  // strlen('int') = 4 <  5, so skip it,
  // even strlen('VeryVeryVeryLongTypeName') > 5.

  unsigned b = static_cast<unsigned>(foo());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name [modernize-use-auto]
  // CHECK-FIXES: auto b = static_cast<unsigned>(foo());

  bool c = static_cast<bool>(foo());
  // strlen('bool') = 4 <  5, so skip it.

  const bool c1 = static_cast<const bool>(foo());
  // strlen('bool') = 4 <  5, so skip it, even there's a 'const'.

  unsigned long long ull = static_cast<unsigned long long>(foo());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name [modernize-use-auto]
  // CHECK-FIXES: auto ull = static_cast<unsigned long long>(foo());

  return 1;
}

