// RUN: %check_clang_tidy -check-suffixes=NORMAL %s cppcoreguidelines-macro-usage %t -- -- -D_ZZZ_IM_A_MACRO
// RUN: %check_clang_tidy -check-suffixes=NORMAL %s cppcoreguidelines-macro-usage %t -- -config='{CheckOptions: [{key: cppcoreguidelines-macro-usage.IgnoreCommandLineMacros, value: 1}]}' -- -D_ZZZ_IM_A_MACRO
// RUN: %check_clang_tidy -check-suffixes=NORMAL,CL %s cppcoreguidelines-macro-usage %t -- -config='{CheckOptions: [{key: cppcoreguidelines-macro-usage.IgnoreCommandLineMacros, value: 0}]}' -- -D_ZZZ_IM_A_MACRO

// CHECK-MESSAGES-CL: warning: macro '_ZZZ_IM_A_MACRO' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT 0
// CHECK-MESSAGES-NORMAL: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT' used to declare a constant; consider using a 'constexpr' constant
