// Remove UNSUPPORTED for powerpc64le when the problem introduced by
// r288563 is resolved.
// UNSUPPORTED: target=powerpc64le{{.*}}
// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.AbstractClassCase: CamelCase, \
// RUN:     readability-identifier-naming.AbstractClassPrefix: 'A', \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassPrefix: 'C', \
// RUN:     readability-identifier-naming.ClassConstantCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassConstantPrefix: 'k', \
// RUN:     readability-identifier-naming.ClassMemberCase: CamelCase, \
// RUN:     readability-identifier-naming.ClassMethodCase: camelBack, \
// RUN:     readability-identifier-naming.ConceptCase: CamelCase, \
// RUN:     readability-identifier-naming.ConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.ConstantSuffix: '_CST', \
// RUN:     readability-identifier-naming.ConstexprFunctionCase: lower_case, \
// RUN:     readability-identifier-naming.ConstexprMethodCase: lower_case, \
// RUN:     readability-identifier-naming.ConstexprVariableCase: lower_case, \
// RUN:     readability-identifier-naming.EnumCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumPrefix: 'E', \
// RUN:     readability-identifier-naming.ScopedEnumConstantCase: CamelCase, \
// RUN:     readability-identifier-naming.EnumConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.FunctionCase: camelBack, \
// RUN:     readability-identifier-naming.GlobalConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.GlobalFunctionCase: CamelCase, \
// RUN:     readability-identifier-naming.GlobalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.GlobalVariablePrefix: 'g_', \
// RUN:     readability-identifier-naming.InlineNamespaceCase: lower_case, \
// RUN:     readability-identifier-naming.LocalConstantCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalConstantPrefix: 'k', \
// RUN:     readability-identifier-naming.LocalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.MemberCase: CamelCase, \
// RUN:     readability-identifier-naming.MemberPrefix: 'm_', \
// RUN:     readability-identifier-naming.ConstantMemberCase: lower_case, \
// RUN:     readability-identifier-naming.PrivateMemberPrefix: '__', \
// RUN:     readability-identifier-naming.ProtectedMemberPrefix: '_', \
// RUN:     readability-identifier-naming.PublicMemberCase: lower_case, \
// RUN:     readability-identifier-naming.MethodCase: camelBack, \
// RUN:     readability-identifier-naming.PrivateMethodPrefix: '__', \
// RUN:     readability-identifier-naming.ProtectedMethodPrefix: '_', \
// RUN:     readability-identifier-naming.NamespaceCase: lower_case, \
// RUN:     readability-identifier-naming.ParameterCase: camelBack, \
// RUN:     readability-identifier-naming.ParameterPrefix: 'a_', \
// RUN:     readability-identifier-naming.ConstantParameterCase: camelBack, \
// RUN:     readability-identifier-naming.ConstantParameterPrefix: 'i_', \
// RUN:     readability-identifier-naming.ParameterPackCase: camelBack, \
// RUN:     readability-identifier-naming.PureFunctionCase: lower_case, \
// RUN:     readability-identifier-naming.PureMethodCase: camelBack, \
// RUN:     readability-identifier-naming.StaticConstantCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.StaticVariableCase: camelBack, \
// RUN:     readability-identifier-naming.StaticVariablePrefix: 's_', \
// RUN:     readability-identifier-naming.StructCase: Leading_upper_snake_case, \
// RUN:     readability-identifier-naming.TemplateParameterCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.TemplateTemplateParameterCase: CamelCase, \
// RUN:     readability-identifier-naming.TemplateUsingCase: lower_case, \
// RUN:     readability-identifier-naming.TemplateUsingPrefix: 'u_', \
// RUN:     readability-identifier-naming.TypeTemplateParameterCase: camelBack, \
// RUN:     readability-identifier-naming.TypeTemplateParameterSuffix: '_t', \
// RUN:     readability-identifier-naming.TypedefCase: lower_case, \
// RUN:     readability-identifier-naming.TypedefSuffix: '_t', \
// RUN:     readability-identifier-naming.UnionCase: CamelCase, \
// RUN:     readability-identifier-naming.UnionPrefix: 'U', \
// RUN:     readability-identifier-naming.UsingCase: lower_case, \
// RUN:     readability-identifier-naming.ValueTemplateParameterCase: camelBack, \
// RUN:     readability-identifier-naming.VariableCase: lower_case, \
// RUN:     readability-identifier-naming.VirtualMethodCase: Camel_Snake_Case, \
// RUN:     readability-identifier-naming.VirtualMethodPrefix: 'v_', \
// RUN:     readability-identifier-naming.MacroDefinitionCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.TypeAliasCase: camel_Snake_Back, \
// RUN:     readability-identifier-naming.TypeAliasSuffix: '_t', \
// RUN:     readability-identifier-naming.IgnoreFailedSplit: false, \
// RUN:     readability-identifier-naming.GlobalPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.GlobalPointerSuffix: '_Ptr', \
// RUN:     readability-identifier-naming.GlobalConstantPointerCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.GlobalConstantPointerSuffix: '_Ptr', \
// RUN:     readability-identifier-naming.PointerParameterCase: lower_case, \
// RUN:     readability-identifier-naming.PointerParameterPrefix: 'p_', \
// RUN:     readability-identifier-naming.ConstantPointerParameterCase: CamelCase, \
// RUN:     readability-identifier-naming.ConstantPointerParameterPrefix: 'cp_', \
// RUN:     readability-identifier-naming.LocalPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalPointerPrefix: 'l_', \
// RUN:     readability-identifier-naming.LocalConstantPointerCase: CamelCase, \
// RUN:     readability-identifier-naming.LocalConstantPointerPrefix: 'lc_', \
// RUN:   }}' -- -fno-delayed-template-parsing -Dbad_macro \
// RUN:   -I%S/Inputs/identifier-naming \
// RUN:   -isystem %S/Inputs/identifier-naming/system

// clang-format off

#include <system-header.h>
#include <coroutines.h>
#include "user-header.h"
// NO warnings or fixes expected from declarations within header files without
// the -header-filter= option

namespace FOO_NS {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for namespace 'FOO_NS' [readability-identifier-naming]
// CHECK-FIXES: {{^}}namespace foo_ns {{{$}}
inline namespace InlineNamespace {
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for inline namespace 'InlineNamespace'
// CHECK-FIXES: {{^}}inline namespace inline_namespace {{{$}}

namespace FOO_ALIAS = FOO_NS;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for namespace 'FOO_ALIAS' [readability-identifier-naming]
// CHECK-FIXES: {{^}}namespace foo_alias = FOO_NS;{{$}}

SYSTEM_NS::structure g_s1;
// NO warnings or fixes expected as SYSTEM_NS and structure are declared in a header file

USER_NS::object g_s2;
// NO warnings or fixes expected as USER_NS and object are declared in a header file

SYSTEM_MACRO(var1);
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'var1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}SYSTEM_MACRO(g_var1);

USER_MACRO(var2);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'var2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}USER_MACRO(g_var2);

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

int g_Global4;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'g_Global4'
// CHECK-FIXES: {{^}}int g_global4;{{$}}

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

enum class EMyEnumeration {
    myConstant = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for scoped enum constant 'myConstant'
// CHECK-FIXES: {{^}}    MyConstant = 1,{{$}}
    your_CONST = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for scoped enum constant 'your_CONST'
// CHECK-FIXES: {{^}}    YourConst = 1,{{$}}
    THIS_ConstValue = 1,
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for scoped enum constant 'THIS_ConstValue'
// CHECK-FIXES: {{^}}    ThisConstValue = 1,{{$}}
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
// No warning needed here as this is tied to the previous declaration.
// Just make sure the fix is applied.
// CHECK-FIXES: {{^}}class CMyClass;{{$}}

class my_forward_declared_class; // No warning should be triggered.

const int my_class::classConstant = 4;
// CHECK-FIXES: {{^}}const int CMyClass::kClassConstant = 4;{{$}}

int my_class::ClassMember_2 = 5;
// CHECK-FIXES: {{^}}int CMyClass::ClassMember2 = 5;{{$}}

class my_derived_class : public virtual my_class {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'my_derived_class'
// CHECK-FIXES: {{^}}class CMyDerivedClass : public virtual CMyClass {};{{$}}

class CMyWellNamedClass {};
// No warning expected as this class is well named.

template<typename t_t>
concept MyConcept = requires (t_t a_t) { {a_t++}; };
// No warning expected as this concept is well named.

template<typename t_t>
concept my_concept_2 = requires (t_t a_t) { {a_t++}; };
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for concept 'my_concept_2'
// CHECK-FIXES: {{^}}concept MyConcept2 = requires (t_t a_t) { {a_t++}; };{{$}}

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

class AOverridden {
public:
  virtual ~AOverridden() = default;
  virtual void BadBaseMethod() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethod'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method() = 0;

  virtual void BadBaseMethodNoAttr() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethodNoAttr'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method_No_Attr() = 0;
};

class COverriding : public AOverridden {
public:
  // Overriding a badly-named base isn't a new violation.
  void BadBaseMethod() override {}
  // CHECK-FIXES: {{^}}  void v_Bad_Base_Method() override {}

  void BadBaseMethodNoAttr() /* override */ {}
  // CHECK-FIXES: {{^}}  void v_Bad_Base_Method_No_Attr() /* override */ {}

  void foo() {
    BadBaseMethod();
    // CHECK-FIXES: {{^}}    v_Bad_Base_Method();
    this->BadBaseMethod();
    // CHECK-FIXES: {{^}}    this->v_Bad_Base_Method();
    AOverridden::BadBaseMethod();
    // CHECK-FIXES: {{^}}    AOverridden::v_Bad_Base_Method();
    COverriding::BadBaseMethod();
    // CHECK-FIXES: {{^}}    COverriding::v_Bad_Base_Method();

    BadBaseMethodNoAttr();
    // CHECK-FIXES: {{^}}    v_Bad_Base_Method_No_Attr();
    this->BadBaseMethodNoAttr();
    // CHECK-FIXES: {{^}}    this->v_Bad_Base_Method_No_Attr();
    AOverridden::BadBaseMethodNoAttr();
    // CHECK-FIXES: {{^}}    AOverridden::v_Bad_Base_Method_No_Attr();
    COverriding::BadBaseMethodNoAttr();
    // CHECK-FIXES: {{^}}    COverriding::v_Bad_Base_Method_No_Attr();
  }
};

// Same test as above, now with a dependent base class.
template<typename some_t>
class ATOverridden {
public:
  virtual void BadBaseMethod() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethod'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method() = 0;

  virtual void BadBaseMethodNoAttr() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethodNoAttr'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method_No_Attr() = 0;
};

template<typename some_t>
class CTOverriding : public ATOverridden<some_t> {
  // Overriding a badly-named base isn't a new violation.
  // FIXME: The fixes from the base class should be propagated to the derived class here
  //        (note that there could be specializations of the template base class, though)
  void BadBaseMethod() override {}

  // Without the "override" attribute, and due to the dependent base class, it is not
  // known whether this method overrides anything, so we get the warning here.
  virtual void BadBaseMethodNoAttr() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethodNoAttr'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method_No_Attr() {};
};

template<typename some_t>
void VirtualCall(AOverridden &a_vItem, ATOverridden<some_t> &a_vTitem) {
  a_vItem.BadBaseMethod();
  // CHECK-FIXES: {{^}}  a_vItem.v_Bad_Base_Method();

  // FIXME: The fixes from ATOverridden should be propagated to the following call
  a_vTitem.BadBaseMethod();
}

// Same test as above, now with a dependent base class that is instantiated below.
template<typename some_t>
class ATIOverridden {
public:
  virtual void BadBaseMethod() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for virtual method 'BadBaseMethod'
  // CHECK-FIXES: {{^}}  virtual void v_Bad_Base_Method() = 0;
};

template<typename some_t>
class CTIOverriding : public ATIOverridden<some_t> {
public:
  // Overriding a badly-named base isn't a new violation.
  void BadBaseMethod() override {}
  // CHECK-FIXES: {{^}}  void v_Bad_Base_Method() override {}
};

template class CTIOverriding<int>;

void VirtualCallI(ATIOverridden<int>& a_vItem, CTIOverriding<int>& a_vCitem) {
  a_vItem.BadBaseMethod();
  // CHECK-FIXES: {{^}}  a_vItem.v_Bad_Base_Method();

  a_vCitem.BadBaseMethod();
  // CHECK-FIXES: {{^}}  a_vCitem.v_Bad_Base_Method();
}

template <typename derived_t>
class CRTPBase {
public:
  void BadBaseMethod(int) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for method 'BadBaseMethod'
};

class CRTPDerived : CRTPBase<CRTPDerived> {
public:
  // Hiding a badly-named base isn't a new violation.
  double BadBaseMethod(double) { return 0; }
};

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
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for type alias 'mysuper_tpl_t'
// CHECK-FIXES: {{^}}using mysuper_Tpl_t = CMyOtherTemplatedClass  <:: foo_ns  ::CMyClass>;{{$}}

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
// CHECK-FIXES: {{^}}struct This_structure {{{$}}
    THIS___Structure();
// CHECK-FIXES: {{^}}    This_structure();{{$}}

  union __MyUnion_is_wonderful__ {};
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for union '__MyUnion_is_wonderful__'
// CHECK-FIXES: {{^}}  union UMyUnionIsWonderful {};{{$}}
};

typedef THIS___Structure struct_type;
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for typedef 'struct_type'
// CHECK-FIXES: {{^}}typedef This_structure struct_type_t;{{$}}

struct_type GlobalTypedefTestFunction(struct_type a_argument1) {
// CHECK-FIXES: {{^}}struct_type_t GlobalTypedefTestFunction(struct_type_t a_argument1) {
    struct_type typedef_test_1;
// CHECK-FIXES: {{^}}    struct_type_t typedef_test_1;
}

using my_struct_type = THIS___Structure;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for type alias 'my_struct_type'
// CHECK-FIXES: {{^}}using my_Struct_Type_t = This_structure;{{$}}

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
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global function 'function' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void MY_TEST_MACRO(Function) {}

#define MY_CAT_IMPL(l, r) l ## r
#define MY_CAT(l, r) MY_CAT_IMPL(l, r)
#define MY_MACRO2(foo) int MY_CAT(awesome_, MY_CAT(foo, __COUNTER__)) = 0
#define MY_MACRO3(foo) int MY_CAT(awesome_, foo) = 0
MY_MACRO2(myglob);
MY_MACRO3(myglob);
// No suggestions should occur even though the resulting decl of awesome_myglob#
// or awesome_myglob are not entirely within a macro argument.

} // namespace InlineNamespace
} // namespace FOO_NS

template <typename t_t> struct a {
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: invalid case style for struct 'a'
// CHECK-FIXES: {{^}}template <typename t_t> struct A {{{$}}
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
// CHECK-FIXES: {{^}}char const A<t_t>::kMyConstClassString[] = "123";{{$}}

template <template <typename> class A> struct b { A<int> c; };
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: invalid case style for struct 'b'
// CHECK-FIXES:template <template <typename> class A> struct B { A<int> c; };{{$}}

unsigned MY_GLOBAL_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'MY_GLOBAL_array'
// CHECK-FIXES: {{^}}unsigned g_my_global_array[] = {1,2,3};{{$}}

unsigned const MyConstGlobal_array[] = {1,2,3};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global constant 'MyConstGlobal_array'
// CHECK-FIXES: {{^}}unsigned const MY_CONST_GLOBAL_ARRAY[] = {1,2,3};{{$}}

int * MyGlobal_Ptr;// -> ok
int * my_second_global_Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'my_second_global_Ptr'
// CHECK-FIXES: {{^}}int * MySecondGlobal_Ptr;{{$}}
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

using namespace FOO_NS;
// CHECK-FIXES: {{^}}using namespace foo_ns;

using namespace FOO_NS::InlineNamespace;
// CHECK-FIXES: {{^}}using namespace foo_ns::inline_namespace;

void QualifiedTypeLocTest(THIS___Structure);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(This_structure);{{$}}
void QualifiedTypeLocTest(THIS___Structure &);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(This_structure &);{{$}}
void QualifiedTypeLocTest(THIS___Structure &&);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(This_structure &&);{{$}}
void QualifiedTypeLocTest(const THIS___Structure);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(const This_structure);{{$}}
void QualifiedTypeLocTest(const THIS___Structure &);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(const This_structure &);{{$}}
void QualifiedTypeLocTest(volatile THIS___Structure &);
// CHECK-FIXES: {{^}}void QualifiedTypeLocTest(volatile This_structure &);{{$}}

namespace redecls {
// We only want the warning to show up once here for the first decl.
// CHECK-MESSAGES: :[[@LINE+1]]:6: warning: invalid case style for global function 'badNamedFunction'
void badNamedFunction();
void badNamedFunction();
void badNamedFunction(){}
//      CHECK-FIXES: {{^}}void BadNamedFunction();
// CHECK-FIXES-NEXT: {{^}}void BadNamedFunction();
// CHECK-FIXES-NEXT: {{^}}void BadNamedFunction(){}
void ReferenceBadNamedFunction() {
  auto l_Ptr = badNamedFunction;
  // CHECK-FIXES: {{^}}  auto l_Ptr = BadNamedFunction;
  l_Ptr();
  badNamedFunction();
  // CHECK-FIXES: {{^}}  BadNamedFunction();
}

} // namespace redecls

namespace scratchspace {
#define DUP(Tok) Tok
#define M1(Tok) DUP(badName##Tok())

// We don't want a warning here as the call to this in Foo is in a scratch
// buffer so its fix-it wouldn't be applied, resulting in invalid code.
void badNameWarn();

void Foo() {
  M1(Warn);
}

#undef M1
#undef DUP
} // namespace scratchspace

template<typename type_t>
auto GetRes(type_t& Param) -> decltype(Param.res());
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for parameter 'Param'
// CHECK-FIXES: auto GetRes(type_t& a_param) -> decltype(a_param.res());

// Check implicit declarations in coroutines

struct async_obj {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'async_obj'
// CHECK-FIXES: {{^}}struct Async_obj {{{$}}
public:
  never_suspend operator co_await() const noexcept;
};

task ImplicitDeclTest(async_obj &a_object) {
  co_await a_object;  // CHECK-MESSAGES-NOT: warning: invalid case style for local variable
}

// Test scenario when canonical declaration will be a forward declaration
struct ForwardDeclStruct;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'ForwardDeclStruct' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct Forward_decl_struct;
// CHECK-FIXES: {{^}}struct Forward_decl_struct {
struct ForwardDeclStruct {
};

struct forward_declared_as_struct;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for class 'forward_declared_as_struct' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct CForwardDeclaredAsStruct;
// CHECK-FIXES: {{^}}class CForwardDeclaredAsStruct {
class forward_declared_as_struct {
};

namespace pr55156 {

template<typename> struct Wrap;

typedef enum {
  VALUE0,
  VALUE1,
} ValueType;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: invalid case style for typedef 'ValueType' [readability-identifier-naming]
// CHECK-FIXES: {{^}}} value_type_t;

typedef ValueType (*MyFunPtr)(const ValueType&, Wrap<ValueType>*);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for typedef 'MyFunPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}typedef value_type_t (*my_fun_ptr_t)(const value_type_t&, Wrap<value_type_t>*);

#define STATIC_MACRO static
STATIC_MACRO void someFunc(ValueType a_v1, const ValueType& a_v2) {}
// CHECK-FIXES: {{^}}STATIC_MACRO void someFunc(value_type_t a_v1, const value_type_t& a_v2) {}
STATIC_MACRO void someFunc(const ValueType** p_a_v1, ValueType (*p_a_v2)()) {}
// CHECK-FIXES: {{^}}STATIC_MACRO void someFunc(const value_type_t** p_a_v1, value_type_t (*p_a_v2)()) {}
STATIC_MACRO ValueType someFunc() {}
// CHECK-FIXES: {{^}}STATIC_MACRO value_type_t someFunc() {}
STATIC_MACRO void someFunc(MyFunPtr, const MyFunPtr****) {}
// CHECK-FIXES: {{^}}STATIC_MACRO void someFunc(my_fun_ptr_t, const my_fun_ptr_t****) {}
#undef STATIC_MACRO
}

struct Some_struct {
  int SomeMember;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for public member 'SomeMember' [readability-identifier-naming]
// CHECK-FIXES: {{^}}  int some_member;
};
Some_struct g_s1{ .SomeMember = 1 };
// CHECK-FIXES: {{^}}Some_struct g_s1{ .some_member = 1 };
Some_struct g_s2{.SomeMember=1};
// CHECK-FIXES: {{^}}Some_struct g_s2{.some_member=1};
