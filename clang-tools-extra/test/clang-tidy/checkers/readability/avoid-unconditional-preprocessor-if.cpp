// RUN: %check_clang_tidy %s readability-avoid-unconditional-preprocessor-if %t

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'false', consider removing both the condition and its contents [readability-avoid-unconditional-preprocessor-if]
#if 0
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'true', consider removing condition but leaving its contents [readability-avoid-unconditional-preprocessor-if]
#if 1
// some code
#endif

#if test

#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'true', consider removing condition but leaving its contents [readability-avoid-unconditional-preprocessor-if]
#if 10>5

#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'false', consider removing both the condition and its contents [readability-avoid-unconditional-preprocessor-if]
#if 10<5

#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'true', consider removing condition but leaving its contents [readability-avoid-unconditional-preprocessor-if]
#if 10 > 5
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'false', consider removing both the condition and its contents [readability-avoid-unconditional-preprocessor-if]
#if 10 < 5
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'false', consider removing both the condition and its contents [readability-avoid-unconditional-preprocessor-if]
#if !(10 > \
        5)
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'true', consider removing condition but leaving its contents [readability-avoid-unconditional-preprocessor-if]
#if !(10 < \
        5)
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'true', consider removing condition but leaving its contents [readability-avoid-unconditional-preprocessor-if]
#if true
// some code
#endif

// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: preprocessor condition is always 'false', consider removing both the condition and its contents [readability-avoid-unconditional-preprocessor-if]
#if false
// some code
#endif

#define MACRO
#ifdef MACRO
// some code
#endif

#if !SOMETHING
#endif

#if !( \
     defined MACRO)
// some code
#endif


#if __has_include(<string_view>)
// some code
#endif

#if __has_include(<string_view_not_exist>)
// some code
#endif

#define DDD 17
#define EEE 18

#if 10 > DDD
// some code
#endif

#if 10 < DDD
// some code
#endif

#if EEE > DDD
// some code
#endif
