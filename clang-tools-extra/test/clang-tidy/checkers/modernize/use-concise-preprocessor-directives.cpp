// RUN: %check_clang_tidy -std=c++98 -check-suffixes=ALL,CXX %s modernize-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++11 -check-suffixes=ALL,CXX %s modernize-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++14 -check-suffixes=ALL,CXX %s modernize-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++17 -check-suffixes=ALL,CXX %s modernize-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++20 -check-suffixes=ALL,CXX %s modernize-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffixes=ALL,23,CXX,CXX23 %s modernize-use-concise-preprocessor-directives %t

// RUN: %check_clang_tidy -std=c99 -check-suffix=ALL %s modernize-use-concise-preprocessor-directives %t -- -- -x c
// RUN: %check_clang_tidy -std=c11 -check-suffix=ALL %s modernize-use-concise-preprocessor-directives %t -- -- -x c
// RUN: %check_clang_tidy -std=c23-or-later -check-suffix=ALL,23 %s modernize-use-concise-preprocessor-directives %t -- -- -x c

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifdef FOO
#if defined(FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined(BAR)
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifdef FOO
#if defined FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined BAR
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifdef FOO
#if (defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:4: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #  elifdef BAR
#  elif (defined(BAR))
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifdef FOO
#if (defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:4: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #  elifdef BAR
#  elif (defined BAR)
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !defined(FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !defined(BAR)
#endif

#ifdef __cplusplus
// CHECK-MESSAGES-CXX: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX: #ifndef FOO
#if not defined(FOO)
// CHECK-MESSAGES-CXX23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX23: #elifndef BAR
#elif not defined(BAR)
#endif
#endif // __cplusplus

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !defined FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !defined BAR
#endif

#ifdef __cplusplus
// CHECK-MESSAGES-CXX: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX: #ifndef FOO
#if not defined FOO
// CHECK-MESSAGES-CXX23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX23: #elifndef BAR
#elif not defined BAR
#endif
#endif // __cplusplus

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if (!defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif (!defined(BAR))
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if (!defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif (!defined BAR)
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !(defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !(defined(BAR))
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !(defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !(defined BAR)
#endif

// These cases with many parentheses and negations are unrealistic, but
// handling them doesn't really add any complexity to the implementation.
// Test them for good measure.

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !((!!(defined(FOO))))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif ((!(!(defined(BAR)))))
#endif

// CHECK-MESSAGES-ALL: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-ALL: #ifndef FOO
#if !((!!(defined FOO)))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely [modernize-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif ((!(!(defined BAR))))
#endif

#if FOO
#elif BAR
#endif
