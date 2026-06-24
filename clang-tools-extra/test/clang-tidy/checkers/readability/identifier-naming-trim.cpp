// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.GlobalVariablePrefix: 'g_', \
// RUN:     readability-identifier-naming.MacroDefinitionPrefix: 'M_', \
// RUN:     readability-identifier-naming.MemberPrefix: 'm_', \
// RUN:     readability-identifier-naming.MemberSuffix: '_', \
// RUN:     readability-identifier-naming.ParameterPrefix: 'p_', \
// RUN:     readability-identifier-naming.TemplateParameterPrefix: t_, \
// RUN:     readability-identifier-naming.TrimPrefixes: 1, \
// RUN:     readability-identifier-naming.TrimSuffixes: 1, \
// RUN:   }}' \
// RUN:   -header-filter='' \
// RUN:   -- -fno-delayed-template-parsing -Dbad_macro \
// RUN:   -I%S/Inputs/identifier-naming \
// RUN:   -isystem %S/Inputs/identifier-naming/system

#include <system-header.h>

#define m_MACRO(arg) void f(int arg) {}
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for macro definition 'm_MACRO'
// CHECK-FIXES: #define M_MACRO(arg) void f(int arg) {}

m_MACRO(m_wrong);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for parameter 'm_wrong'
// CHECK-FIXES: M_MACRO(p_wrong);

SYSTEM_MACRO(m_wrong);
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'm_wrong'
// CHECK-FIXES: SYSTEM_MACRO(g_wrong);

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

template<typename p_t>
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for template parameter 'p_t'
// CHECK-FIXES: template<typename t_t>
void f(p_t p_arg) {}
// CHECK-FIXES: void f(t_t p_arg) {}
