// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.MemberPrefix: 'm_', \
// RUN:     readability-identifier-naming.MemberSuffix: '_', \
// RUN:     readability-identifier-naming.ParameterPrefix: 'p_', \
// RUN:     readability-identifier-naming.TrimPrefixes: 1, \
// RUN:     readability-identifier-naming.TrimSuffixes: 1, \
// RUN:   }}' \
// RUN:   -header-filter='' \
// RUN:   -- -fno-delayed-template-parsing -Dbad_macro \
// RUN:   -I%S/Inputs/identifier-naming \
// RUN:   -isystem %S/Inputs/identifier-naming/system

// clang-format off

struct Triple {
    Triple(int m_wrong_, int missing, int p_ok): p_wrong_(m_wrong_), missing(missing), m_ok_(p_ok) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for parameter 'm_wrong_'
    // CHECK-MESSAGES: :[[@LINE-2]]:30: warning: invalid case style for parameter 'missing'
    // CHECK-FIXES: Triple(int p_wrong, int p_missing, int p_ok): m_wrong_(p_wrong), m_missing_(p_missing), m_ok_(p_ok) {}
    int p_wrong_;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for member 'p_wrong_'
    // CHECK-FIXES: int m_wrong_;
    int missing;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for member 'missing'
    // CHECK-FIXES: int m_missing_;
    int m_ok_;
};

void multipleWrong(int m_m_wrong1, int wrong2__) {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for parameter 'm_m_wrong1'
// CHECK-MESSAGES: :[[@LINE-2]]:40: warning: invalid case style for parameter 'wrong2__'
// CHECK-FIXES: void multipleWrong(int p_wrong1, int p_wrong2) {}
