// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-use-enum-class %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-use-enum-class.IgnoreMacros: false}}"
// RUN: %check_clang_tidy -std=c++11-or-later %s -check-suffixes=IGNORE-MACROS cppcoreguidelines-use-enum-class %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-use-enum-class.IgnoreMacros: true}}"

enum UnscopedRegular { A, B, C };
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'UnscopedRegular' is unscoped, use 'enum class' instead
// CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE-2]]:6: warning: enum 'UnscopedRegular' is unscoped, use 'enum class' instead

#define NAMED_ENUM enum E { G, H, I };

NAMED_ENUM
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: enum 'E' is unscoped, use 'enum class' instead
