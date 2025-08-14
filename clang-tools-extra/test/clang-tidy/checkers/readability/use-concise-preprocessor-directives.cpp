// RUN: %check_clang_tidy -std=c++98,c++11,c++14,c++17,c++20 -check-suffixes=,CXX %s readability-use-concise-preprocessor-directives %t
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffixes=,23,CXX,CXX23 %s readability-use-concise-preprocessor-directives %t

// RUN: %check_clang_tidy -std=c99,c11,c17 %s readability-use-concise-preprocessor-directives %t -- -- -x c
// RUN: %check_clang_tidy -std=c23-or-later -check-suffixes=,23 %s readability-use-concise-preprocessor-directives %t -- -- -x c

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if defined(FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined(BAR)
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if defined FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined BAR
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if (defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:4: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #  elifdef BAR
#  elif (defined(BAR))
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if (defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:4: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #  elifdef BAR
#  elif (defined BAR)
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !defined(FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !defined(BAR)
#endif

#ifdef __cplusplus
// CHECK-MESSAGES-CXX: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX: #ifndef FOO
#if not defined(FOO)
// CHECK-MESSAGES-CXX23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX23: #elifndef BAR
#elif not defined(BAR)
#endif
#endif // __cplusplus

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !defined FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !defined BAR
#endif

#ifdef __cplusplus
// CHECK-MESSAGES-CXX: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX: #ifndef FOO
#if not defined FOO
// CHECK-MESSAGES-CXX23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-CXX23: #elifndef BAR
#elif not defined BAR
#endif
#endif // __cplusplus

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if (!defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif (!defined(BAR))
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if (!defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif (!defined BAR)
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !(defined(FOO))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !(defined(BAR))
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !(defined FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !(defined BAR)
#endif

// These cases with many parentheses and negations are unrealistic, but
// handling them doesn't really add any complexity to the implementation.
// Test them for good measure.

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !((!!(defined(FOO))))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif ((!(!(defined(BAR)))))
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !((!!(defined FOO)))
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif ((!(!(defined BAR))))
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !(  (!! ( defined FOO  )) )
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif   ( (  !(!( defined   BAR) )  ))
#endif

#if FOO
#elif BAR
#endif

#if defined(FOO) && defined(BAR)
#elif defined(FOO) && defined(BAR)
#endif

#if defined FOO && BAR
#endif
