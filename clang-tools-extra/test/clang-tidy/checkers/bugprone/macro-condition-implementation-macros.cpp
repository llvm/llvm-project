// RUN: clang-tidy %s -checks=-*,bugprone-macro-condition -- | count 0
// RUN: %check_clang_tidy -check-suffix=ENABLED %s \
// RUN:   bugprone-macro-condition %t -- \
// RUN:   -config="{CheckOptions: {bugprone-macro-condition.CheckDoubleUnderscoreMacros: true}}"

#define __IMPLEMENTATION_VERSION__ 1200

#ifdef __IMPLEMENTATION_VERSION__
#endif

#if __IMPLEMENTATION_VERSION__ >= 1200
// CHECK-MESSAGES-ENABLED: :[[@LINE-1]]:5: warning: Macro '__IMPLEMENTATION_VERSION__' checked here for value after being checked for definition
// CHECK-MESSAGES-ENABLED: :[[@LINE-5]]:2: note: Macro '__IMPLEMENTATION_VERSION__' first checked here for definition
#endif
