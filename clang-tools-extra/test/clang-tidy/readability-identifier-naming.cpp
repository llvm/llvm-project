// Remove UNSUPPORTED for powerpc64le when the problem introduced by
// r288563 is resolved.
// UNSUPPORTED: powerpc64le
// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: readability-identifier-naming.AbstractClassCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.AbstractClassPrefix, value: 'A'}, \
// RUN:     {key: readability-identifier-naming.ClassCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.ClassPrefix, value: 'C'}, \
// RUN:     {key: readability-identifier-naming.ClassConstantCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.ClassConstantPrefix, value: 'k'}, \
// RUN:     {key: readability-identifier-naming.ClassMemberCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.ClassMethodCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.ConstantCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.ConstantSuffix, value: '_CST'}, \
// RUN:     {key: readability-identifier-naming.ConstexprFunctionCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.ConstexprMethodCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.ConstexprVariableCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.EnumCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.EnumPrefix, value: 'E'}, \
// RUN:     {key: readability-identifier-naming.EnumConstantCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.FunctionCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.GlobalConstantCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.GlobalFunctionCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.GlobalVariableCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.GlobalVariablePrefix, value: 'g_'}, \
// RUN:     {key: readability-identifier-naming.InlineNamespaceCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.LocalConstantCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.LocalConstantPrefix, value: 'k'}, \
// RUN:     {key: readability-identifier-naming.LocalVariableCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.MemberCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.MemberPrefix, value: 'm_'}, \
// RUN:     {key: readability-identifier-naming.ConstantMemberCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.PrivateMemberPrefix, value: '__'}, \
// RUN:     {key: readability-identifier-naming.ProtectedMemberPrefix, value: '_'}, \
// RUN:     {key: readability-identifier-naming.PublicMemberCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.MethodCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.PrivateMethodPrefix, value: '__'}, \
// RUN:     {key: readability-identifier-naming.ProtectedMethodPrefix, value: '_'}, \
// RUN:     {key: readability-identifier-naming.NamespaceCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.ParameterCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.ParameterPrefix, value: 'a_'}, \
// RUN:     {key: readability-identifier-naming.ConstantParameterCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.ConstantParameterPrefix, value: 'i_'}, \
// RUN:     {key: readability-identifier-naming.ParameterPackCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.PureFunctionCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.PureMethodCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.StaticConstantCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.StaticVariableCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.StaticVariablePrefix, value: 's_'}, \
// RUN:     {key: readability-identifier-naming.StructCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.TemplateParameterCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.TemplateTemplateParameterCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.TemplateUsingCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.TemplateUsingPrefix, value: 'u_'}, \
// RUN:     {key: readability-identifier-naming.TypeTemplateParameterCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.TypeTemplateParameterSuffix, value: '_t'}, \
// RUN:     {key: readability-identifier-naming.TypedefCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.TypedefSuffix, value: '_t'}, \
// RUN:     {key: readability-identifier-naming.UnionCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.UnionPrefix, value: 'U'}, \
// RUN:     {key: readability-identifier-naming.UsingCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.ValueTemplateParameterCase, value: camelBack}, \
// RUN:     {key: readability-identifier-naming.VariableCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.VirtualMethodCase, value: Camel_Snake_Case}, \
// RUN:     {key: readability-identifier-naming.VirtualMethodPrefix, value: 'v_'}, \
// RUN:     {key: readability-identifier-naming.MacroDefinitionCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.TypeAliasCase, value: camel_Snake_Back}, \
// RUN:     {key: readability-identifier-naming.TypeAliasSuffix, value: '_t'}, \
// RUN:     {key: readability-identifier-naming.IgnoreFailedSplit, value: 0}, \
// RUN:     {key: readability-identifier-naming.GlobalPointerCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.GlobalPointerSuffix, value: '_Ptr'}, \
// RUN:     {key: readability-identifier-naming.GlobalConstantPointerCase, value: UPPER_CASE}, \
// RUN:     {key: readability-identifier-naming.GlobalConstantPointerSuffix, value: '_Ptr'}, \
// RUN:     {key: readability-identifier-naming.PointerParameterCase, value: lower_case}, \
// RUN:     {key: readability-identifier-naming.PointerParameterPrefix, value: 'p_'}, \
// RUN:     {key: readability-identifier-naming.ConstantPointerParameterCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.ConstantPointerParameterPrefix, value: 'cp_'}, \
// RUN:     {key: readability-identifier-naming.LocalPointerCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.LocalPointerPrefix, value: 'l_'}, \
// RUN:     {key: readability-identifier-naming.LocalConstantPointerCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.LocalConstantPointerPrefix, value: 'lc_'}, \
// RUN:   ]}' -- -std=c++11 -fno-delayed-template-parsing \
// RUN:   -I%S/Inputs/readability-identifier-naming \
// RUN:   -isystem %S/Inputs/readability-identifier-naming/system

// clang-format off

#include <system-header.h>
#include "user-header.h"
// NO warnings or fixes expected from declarations within header files without
// the -header-filter= option

namespace FOO_NS {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for namespace 'FOO_NS' [readability-identifier-naming]
// CHECK-FIXES: {{^}}namespace foo_ns {{{$}}
inline namespace InlineNamespace {
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for inline namespace 'InlineNamespace'
// CHECK-FIXES: {{^}}inline namespace inline_namespace {{{$}}

SYSTEM_NS::structure g_s1;
// NO warnings or fixes expected as SYSTEM_NS and structure are declared in a header file

USER_NS::object g_s2;
// NO warnings or fixes expected as USER_NS and object are declared in a header file

SYSTEM_MACRO(var1);
// NO warnings or fixes expected as var1 is from macro expansion

USER_MACRO(var2);
// NO warnings or fixes expected as var2 is declared in a macro expansion

#define BLA int FOO_bar
BLA;
// NO warnings or fixes expected as FOO_bar is from macro expansion

int global0;
#define USE_NUMBERED_GLOBAL(number) auto use_global##number = global##number
USE_NUMBERED_GLOBAL(0);
// NO warnings or fixes expected as global0 is pieced together in a macro
// expansion.

int global1;
#define USE_NUMBERED_BAL(prefix, number) \
  auto use_##prefix##bal##number = prefix##bal##number
USE_NUMBERED_BAL(glo, 1);
// NO warnings or fixes expected as global1 is pieced together in a macro
// expansion.

int global2;
#define USE_RECONSTRUCTED(glo, bal) auto use_##glo##bal = glo##bal
USE_RECONSTRUCTED(glo, bal2);
// NO warnings or fixes expected as global2 is pieced together in a macro
// expansion.

int global;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'global'
// CHECK-FIXES: {{^}}int g_global;{{$}}
#define USE_IN_MACRO(m) auto use_##m = m
USE_IN_MACRO(global);

int global3;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'global3'
// CHECK-FIXES: {{^}}int g_global3;{{$}}
#define ADD_TO_SELF(m) (m) + (m)
int g_twice_global3 = ADD_TO_SELF(global3);
// CHECK-FIXES: {{^}}int g_twice_global3 = ADD_TO_SELF(g_global3);{{$}}

enum my_enumeration {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for enum 'my_enumeration'
// CHECK-FIXES: {{^}}enum EMyEnumeration {{{$}}
    MyConstant = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for enum constant 'MyConstant'
// CHECK-FIXES: {{^}}    MY_CONSTANT = 1,{{$}}
    your_CONST = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for enum constant 'your_CONST'
// CHECK-FIXES: {{^}}    YOUR_CONST = 1,{{$}}
    THIS_ConstValue = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for enum constant 'THIS_ConstValue'
// CHECK-FIXES: {{^}}    THIS_CONST_VALUE = 1,{{$}}
};

constexpr int ConstExpr_variable = MyConstant;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for constexpr variable 'ConstExpr_variable'
// CHECK-FIXES: {{^}}constexpr int const_expr_variable = MY_CONSTANT;{{$}}

class my_class {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_class'
// CHECK-FIXES: {{^}}class CMyClass {{{$}}
public:
    my_class();
// CHECK-FIXES: {{^}}    CMyClass();{{$}}

    my_class(void*) : my_class() {}
// CHECK-FIXES: {{^}}    CMyClass(void*) : CMyClass() {}{{$}}

    ~
      my_class();
// (space in destructor token test, we could check trigraph but they will be deprecated)
// CHECK-FIXES: {{^}}    ~{{$}}
// CHECK-FIXES: {{^}}      CMyClass();{{$}}

private:
  const int MEMBER_one_1 = ConstExpr_variable;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for constant member 'MEMBER_one_1'
// CHECK-FIXES: {{^}}  const int member_one_1 = const_expr_variable;{{$}}
  int member2 = 2;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for private member 'member2'
// CHECK-FIXES: {{^}}  int __member2 = 2;{{$}}
  int _memberWithExtraUnderscores_ = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for private member '_memberWithExtraUnderscores_'
// CHECK-FIXES: {{^}}  int __memberWithExtraUnderscores = 42;{{$}}

private:
    int private_member = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for private member 'private_member'
// CHECK-FIXES: {{^}}    int __private_member = 3;{{$}}

protected:
    int ProtMember;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for protected member 'ProtMember'
// CHECK-FIXES: {{^}}    int _ProtMember;{{$}}

public:
    int PubMem;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for public member 'PubMem'
// CHECK-FIXES: {{^}}    int pub_mem;{{$}}

    static const int classConstant;
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: invalid case style for class constant 'classConstant'
// CHECK-FIXES: {{^}}    static const int kClassConstant;{{$}}
    static int ClassMember_2;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for class member 'ClassMember_2'
// CHECK-FIXES: {{^}}    static int ClassMember2;{{$}}
};
class my_class;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_class'
// CHECK-FIXES: {{^}}class CMyClass;{{$}}

class my_forward_declared_class; // No warning should be triggered.

const int my_class::classConstant = 4;
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for class constant 'classConstant'
// CHECK-FIXES: {{^}}const int CMyClass::kClassConstant = 4;{{$}}

int my_class::ClassMember_2 = 5;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for class member 'ClassMember_2'
// CHECK-FIXES: {{^}}int CMyClass::ClassMember2 = 5;{{$}}

class my_derived_class : public virtual my_class {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_derived_class'
// CHECK-FIXES: {{^}}class CMyDerivedClass : public virtual CMyClass {};{{$}}

class CMyWellNamedClass {};
// No warning expected as this class is well named.

template <typename t_t>
class CMyWellNamedClass2 : public my_class {
  // CHECK-FIXES: {{^}}class CMyWellNamedClass2 : public CMyClass {{{$}}
  t_t my_Bad_Member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for private member 'my_Bad_Member'
  // CHECK-FIXES: {{^}}  t_t __my_Bad_Member;{{$}}
  int my_Other_Bad_Member = 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for private member 'my_Other_Bad_Member'
  // CHECK-FIXES: {{^}}  int __my_Other_Bad_Member = 42;{{$}}
public:
  CMyWellNamedClass2() = default;
  CMyWellNamedClass2(CMyWellNamedClass2 const&) = default;
  CMyWellNamedClass2(CMyWellNamedClass2 &&) = default;
  CMyWellNamedClass2(t_t a_v, void *p_p) : my_class(p_p), my_Bad_Member(a_v) {}
  // CHECK-FIXES: {{^}}  CMyWellNamedClass2(t_t a_v, void *p_p) : CMyClass(p_p), __my_Bad_Member(a_v) {}{{$}}

  CMyWellNamedClass2(t_t a_v) : my_class(), my_Bad_Member(a_v), my_Other_Bad_Member(11) {}
  // CHECK-FIXES: {{^}}  CMyWellNamedClass2(t_t a_v) : CMyClass(), __my_Bad_Member(a_v), __my_Other_Bad_Member(11) {}{{$}}
};
void InstantiateClassMethods() {
  // Ensure we trigger the instantiation of each constructor
  CMyWellNamedClass2<int> x;
  CMyWellNamedClass2<int> x2 = x;
  CMyWellNamedClass2<int> x3 = static_cast<CMyWellNamedClass2<int>&&>(x2);
  CMyWellNamedClass2<int> x4(42);
  CMyWellNamedClass2<int> x5(42, nullptr);
}

template<typename T>
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for type template parameter 'T'
// CHECK-FIXES: {{^}}template<typename t_t>{{$}}
class my_templated_class : CMyWellNamedClass {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_templated_class'
// CHECK-FIXES: {{^}}class CMyTemplatedClass : CMyWellNamedClass {};{{$}}

template<typename T>
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for type template parameter 'T'
// CHECK-FIXES: {{^}}template<typename t_t>{{$}}
class my_other_templated_class : my_templated_class<  my_class>, private my_derived_class {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_other_templated_class'
// CHECK-FIXES: {{^}}class CMyOtherTemplatedClass : CMyTemplatedClass<  CMyClass>, private CMyDerivedClass {};{{$}}

template<typename t_t>
using mysuper_tpl_t = my_other_templated_class  <:: FOO_NS  ::my_class>;
// CHECK-FIXES: {{^}}using mysuper_tpl_t = CMyOtherTemplatedClass  <:: foo_ns  ::CMyClass>;{{$}}

const int global_Constant = 6;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global constant 'global_Constant'
// CHECK-FIXES: {{^}}const int GLOBAL_CONSTANT = 6;{{$}}
int Global_variable = 7;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Global_variable'
// CHECK-FIXES: {{^}}int g_global_variable = 7;{{$}}

void global_function(int PARAMETER_1, int const CONST_parameter) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global function 'global_function'
// CHECK-MESSAGES: :[[@LINE-2]]:26: warning: invalid case style for parameter 'PARAMETER_1'
// CHECK-MESSAGES: :[[@LINE-3]]:49: warning: invalid case style for constant parameter 'CONST_parameter'
// CHECK-FIXES: {{^}}void GlobalFunction(int a_parameter1, int const i_constParameter) {{{$}}
    static const int THIS_static_ConsTant = 4;
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: invalid case style for static constant 'THIS_static_ConsTant'
// CHECK-FIXES: {{^}}    static const int THIS_STATIC_CONS_TANT = 4;{{$}}
    static int THIS_static_variable;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for static variable 'THIS_static_variable'
// CHECK-FIXES: {{^}}    static int s_thisStaticVariable;{{$}}
    int const local_Constant = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for local constant 'local_Constant'
// CHECK-FIXES: {{^}}    int const kLocalConstant = 3;{{$}}
    int LOCAL_VARIABLE;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for local variable 'LOCAL_VARIABLE'
// CHECK-FIXES: {{^}}    int local_variable;{{$}}

    int LOCAL_Array__[] = {0, 1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for local variable 'LOCAL_Array__'
// CHECK-FIXES: {{^}}    int local_array[] = {0, 1, 2};{{$}}

    for (auto _ : LOCAL_Array__) {
    }
}

template<typename ... TYPE_parameters>
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: invalid case style for type template parameter 'TYPE_parameters'
// CHECK-FIXES: {{^}}template<typename ... typeParameters_t>{{$}}
void Global_Fun(TYPE_parameters... PARAMETER_PACK) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global function 'Global_Fun'
// CHECK-MESSAGES: :[[@LINE-2]]:36: warning: invalid case style for parameter pack 'PARAMETER_PACK'
// CHECK-FIXES: {{^}}void GlobalFun(typeParameters_t... parameterPack) {{{$}}
    global_function(1, 2);
// CHECK-FIXES: {{^}}    GlobalFunction(1, 2);{{$}}
    FOO_bar = Global_variable;
// CHECK-FIXES: {{^}}    FOO_bar = g_global_variable;{{$}}
// NO fix expected for FOO_bar declared in macro expansion
}

template<template<typename> class TPL_parameter, int COUNT_params, typename ... TYPE_parameters>
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: invalid case style for template template parameter 'TPL_parameter'
// CHECK-MESSAGES: :[[@LINE-2]]:54: warning: invalid case style for value template parameter 'COUNT_params'
// CHECK-MESSAGES: :[[@LINE-3]]:81: warning: invalid case style for type template parameter 'TYPE_parameters'
// CHECK-FIXES: {{^}}template<template<typename> class TplParameter, int countParams, typename ... typeParameters_t>{{$}}
class test_CLASS {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'test_CLASS'
// CHECK-FIXES: {{^}}class CTestClass {{{$}}
};

class abstract_class {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'abstract_class'
// CHECK-FIXES: {{^}}class AAbstractClass {{{$}}
    virtual ~abstract_class() = 0;
// CHECK-FIXES: {{^}}    virtual ~AAbstractClass() = 0;{{$}}
    virtual void VIRTUAL_METHOD();
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for virtual method 'VIRTUAL_METHOD'
// CHECK-FIXES: {{^}}    virtual void v_Virtual_Method();{{$}}
    void non_Virtual_METHOD() {}
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for private method 'non_Virtual_METHOD'
// CHECK-FIXES: {{^}}    void __non_Virtual_METHOD() {}{{$}}

public:
    static void CLASS_METHOD() {}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for class method 'CLASS_METHOD'
// CHECK-FIXES: {{^}}    static void classMethod() {}{{$}}

    constexpr int CST_expr_Method() { return 2; }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for constexpr method 'CST_expr_Method'
// CHECK-FIXES: {{^}}    constexpr int cst_expr_method() { return 2; }{{$}}

private:
    void PRIVate_Method();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for private method 'PRIVate_Method'
// CHECK-FIXES: {{^}}    void __PRIVate_Method();{{$}}
protected:
    void protected_Method();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for protected method 'protected_Method'
// CHECK-FIXES: {{^}}    void _protected_Method();{{$}}
public:
    void public_Method();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for method 'public_Method'
// CHECK-FIXES: {{^}}    void publicMethod();{{$}}
};

constexpr int CE_function() { return 3; }
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for constexpr function 'CE_function'
// CHECK-FIXES: {{^}}constexpr int ce_function() { return 3; }{{$}}

struct THIS___Structure {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'THIS___Structure'
// CHECK-FIXES: {{^}}struct this_structure {{{$}}
    THIS___Structure();
// CHECK-FIXES: {{^}}    this_structure();{{$}}

  union __MyUnion_is_wonderful__ {};
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for union '__MyUnion_is_wonderful__'
// CHECK-FIXES: {{^}}  union UMyUnionIsWonderful {};{{$}}
};

typedef THIS___Structure struct_type;
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for typedef 'struct_type'
// CHECK-FIXES: {{^}}typedef this_structure struct_type_t;{{$}}

struct_type GlobalTypedefTestFunction(struct_type a_argument1) {
// CHECK-FIXES: {{^}}struct_type_t GlobalTypedefTestFunction(struct_type_t a_argument1) {
    struct_type typedef_test_1;
// CHECK-FIXES: {{^}}    struct_type_t typedef_test_1;
}

using my_struct_type = THIS___Structure;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for type alias 'my_struct_type'
// CHECK-FIXES: {{^}}using my_Struct_Type_t = this_structure;{{$}}

template<typename t_t>
using SomeOtherTemplate = my_other_templated_class  <:: FOO_NS  ::my_class>;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for type alias 'SomeOtherTemplate'
// CHECK-FIXES: {{^}}using some_Other_Template_t = CMyOtherTemplatedClass  <:: foo_ns  ::CMyClass>;{{$}}

static void static_Function() {
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for function 'static_Function'
// CHECK-FIXES: {{^}}static void staticFunction() {{{$}}

  ::FOO_NS::InlineNamespace::abstract_class::CLASS_METHOD();
// CHECK-FIXES: {{^}}  ::foo_ns::inline_namespace::AAbstractClass::classMethod();{{$}}
  ::FOO_NS::InlineNamespace::static_Function();
// CHECK-FIXES: {{^}}  ::foo_ns::inline_namespace::staticFunction();{{$}}

  using ::FOO_NS::InlineNamespace::CE_function;
// CHECK-FIXES: {{^}}  using ::foo_ns::inline_namespace::ce_function;{{$}}

  unsigned MY_LOCAL_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for local variable 'MY_LOCAL_array'
// CHECK-FIXES: {{^}}  unsigned my_local_array[] = {1,2,3};{{$}}

  unsigned const MyConstLocal_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for local constant 'MyConstLocal_array'
// CHECK-FIXES: {{^}}  unsigned const kMyConstLocalArray[] = {1,2,3};{{$}}

  static unsigned MY_STATIC_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for static variable 'MY_STATIC_array'
// CHECK-FIXES: {{^}}  static unsigned s_myStaticArray[] = {1,2,3};{{$}}

  static unsigned const MyConstStatic_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for static constant 'MyConstStatic_array'
// CHECK-FIXES: {{^}}  static unsigned const MY_CONST_STATIC_ARRAY[] = {1,2,3};{{$}}

  char MY_LOCAL_string[] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for local variable 'MY_LOCAL_string'
// CHECK-FIXES: {{^}}  char my_local_string[] = "123";{{$}}

  char const MyConstLocal_string[] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for local constant 'MyConstLocal_string'
// CHECK-FIXES: {{^}}  char const kMyConstLocalString[] = "123";{{$}}

  static char MY_STATIC_string[] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for static variable 'MY_STATIC_string'
// CHECK-FIXES: {{^}}  static char s_myStaticString[] = "123";{{$}}

  static char const MyConstStatic_string[] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for static constant 'MyConstStatic_string'
// CHECK-FIXES: {{^}}  static char const MY_CONST_STATIC_STRING[] = "123";{{$}}
}

#define MY_TEST_Macro(X) X()
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for macro definition 'MY_TEST_Macro'
// CHECK-FIXES: {{^}}#define MY_TEST_MACRO(X) X()

void MY_TEST_Macro(function) {}
// CHECK-FIXES: {{^}}void MY_TEST_MACRO(function) {}
}
}

template <typename t_t> struct a {
  typename t_t::template b<> c;

  char const MY_ConstMember_string[4] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for constant member 'MY_ConstMember_string'
// CHECK-FIXES: {{^}}  char const my_const_member_string[4] = "123";{{$}}

  static char const MyConstClass_string[];
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for class constant 'MyConstClass_string'
// CHECK-FIXES: {{^}}  static char const kMyConstClassString[];{{$}}
};

template<typename t_t>
char const a<t_t>::MyConstClass_string[] = "123";
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for class constant 'MyConstClass_string'
// CHECK-FIXES: {{^}}char const a<t_t>::kMyConstClassString[] = "123";{{$}}

template <template <typename> class A> struct b { A<int> c; };

unsigned MY_GLOBAL_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'MY_GLOBAL_array'
// CHECK-FIXES: {{^}}unsigned g_my_global_array[] = {1,2,3};{{$}}

unsigned const MyConstGlobal_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global constant 'MyConstGlobal_array'
// CHECK-FIXES: {{^}}unsigned const MY_CONST_GLOBAL_ARRAY[] = {1,2,3};{{$}}

int * MyGlobal_Ptr;// -> ok
int * const MyConstantGlobalPointer = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global constant pointer 'MyConstantGlobalPointer'
// CHECK-FIXES: {{^}}int * const MY_CONSTANT_GLOBAL_POINTER_Ptr = nullptr;{{$}}

void MyPoiterFunction(int * p_normal_pointer, int * const constant_ptr){
// CHECK-MESSAGES: :[[@LINE-1]]:59: warning: invalid case style for constant pointer parameter 'constant_ptr'
// CHECK-FIXES: {{^}}void MyPoiterFunction(int * p_normal_pointer, int * const cp_ConstantPtr){{{$}}
    int * l_PointerA;
    int * const pointer_b = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for local constant pointer 'pointer_b'
// CHECK-FIXES: {{^}}    int * const lc_PointerB = nullptr;{{$}}
}

