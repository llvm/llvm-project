// RUN: %check_clang_tidy -std=c++98,c++11,c++14,c++17,c++20 %s readability-use-concise-preprocessor-directives %t -- \
// RUN:   -config='{ CheckOptions: {readability-use-concise-preprocessor-directives.PreserveConsistency: true} }'
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffixes=,23 %s readability-use-concise-preprocessor-directives %t -- \
// RUN:   -config='{ CheckOptions: {readability-use-concise-preprocessor-directives.PreserveConsistency: true} }'

// RUN: %check_clang_tidy -std=c99,c11,c17 %s readability-use-concise-preprocessor-directives %t -- \
// RUN:   -config='{ CheckOptions: {readability-use-concise-preprocessor-directives.PreserveConsistency: true} }' -- -x c
// RUN: %check_clang_tidy -std=c23-or-later -check-suffixes=,23 %s readability-use-concise-preprocessor-directives %t -- \
// RUN:   -config='{ CheckOptions: {readability-use-concise-preprocessor-directives.PreserveConsistency: true} }' -- -x c

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef HEADER_GUARD
#if !defined(HEADER_GUARD)
#define HEADER_GUARD

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if defined(FOO)
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifndef FOO
#if !defined(FOO)
#endif

// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #ifndef FOO
#if !defined(FOO)
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined(BAR)
#endif

// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #ifdef FOO
#if defined FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined BAR
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAZ
#elif !defined BAZ
#endif

#ifdef FOO
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifdef BAR
#elif defined BAR
#endif

#if (     defined(__cplusplus) &&      __cplusplus >= 202302L) || \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)


// Existing code can't decide between concise and verbose form, but
// we can rewrite it to be consistent.
//
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #ifndef FOO
#if !defined(FOO)
#elifdef BAR
#endif

#endif

// Existing code can't decide between concise and verbose form, and rewriting 
// the '#elif defined(BAR)' won't make it more consistent, so leave it alone.
#ifdef FOO
#elif defined(BAR)
#elif defined(BAZ) || defined(HAZ)
#endif

#if defined(FOO)
#elif defined(BAR) || defined(BAZ)
#endif

#if defined(FOO)
#elif defined(BAR) && BAR == 0xC0FFEE 
#endif

#if defined(FOO)
#elif BAR == 10 || defined(BAZ) 
#endif

#if FOO
#elif BAR
#endif

#if defined FOO && BAR
#elif defined BAZ
#endif

// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES: #ifdef FOO
#if defined(FOO)
#elif BAR
#endif

// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#ifdef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #ifdef FOO
#if defined(FOO)
#elif 1 == 1
// CHECK-MESSAGES-23: :[[@LINE+2]]:2: warning: preprocessor condition can be written more concisely using '#elifndef' [readability-use-concise-preprocessor-directives]
// CHECK-FIXES-23: #elifndef BAR
#elif !defined(BAR)
#endif

#endif // HEADER_GUARD
