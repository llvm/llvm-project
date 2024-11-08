// RUN: rm -rf %t
// RUN: not %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -std=c++14 -ast-dump-lookups 2>/dev/null | FileCheck %s --check-prefix=CHECK-GLOBAL
// RUN: not %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -std=c++14 -ast-dump-lookups -ast-dump-filter N 2>/dev/null | FileCheck %s --check-prefix=CHECK-NAMESPACE-N
// RUN: not %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -std=c++14 -ast-dump -ast-dump-filter SomeTemplate 2>/dev/null | FileCheck %s --check-prefix=CHECK-DUMP
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++14
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++14 -DEARLY_IMPORT

#ifdef EARLY_IMPORT
#include "cxx-templates-textual.h"
#endif

@import cxx_templates_a;
@import cxx_templates_b;
@import cxx_templates_c;
@import cxx_templates_d;
@import cxx_templates_common;

template<typename, char> struct Tmpl_T_C {};
template<typename, int, int> struct Tmpl_T_I_I {};

template<typename A, typename B, A> struct Tmpl_T_T_A {};
template<typename A, typename B, B> struct Tmpl_T_T_B {};

template<int> struct UseInt {};

void g() {
  f(0);
  f<double>(1.0);
  f<int>();
  f(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-a.h:3 {{couldn't infer template argument}}
  // expected-note@Inputs/cxx-templates-a.h:4 {{requires 1 argument}}

  N::f(0);
  N::f<double>(1.0);
  N::f<int>();
  N::f(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-b.h:6 {{couldn't infer template argument}}
  // expected-note@Inputs/cxx-templates-b.h:7 {{requires single argument}}

  template_param_kinds_1<0>(); // ok, from cxx-templates-a.h
  template_param_kinds_1<int>(); // ok, from cxx-templates-b.h

  template_param_kinds_2<Tmpl_T_C>(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-a.h:11 {{invalid explicitly-specified argument}}
  // expected-note@Inputs/cxx-templates-b.h:11 {{invalid explicitly-specified argument}}

  template_param_kinds_2<Tmpl_T_I_I>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:11 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:11 {{candidate}}

  // FIXME: This should be valid, but we incorrectly match the template template
  // argument against both template template parameters.
  template_param_kinds_3<Tmpl_T_T_A>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:12 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:12 {{candidate}}
  template_param_kinds_3<Tmpl_T_T_B>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:12 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:12 {{candidate}}

  // Trigger the instantiation of a template in 'a' that uses a type defined in
  // 'common'. That type is not visible here.
  PerformDelayedLookup(defined_in_common);

  // Likewise, but via a default argument.
  PerformDelayedLookupInDefaultArgument(defined_in_common);

  // Trigger the instantiation of a template in 'b' that uses a type defined in
  // 'b_impl'. That type is not visible here.
  UseDefinedInBImpl<int>();

  // Trigger the instantiation of a template in 'a' that uses a type defined in
  // 'b_impl', via a template defined in 'b'. Since the type is visible from
  // within 'b', the instantiation succeeds.
  UseDefinedInBImplIndirectly(defined_in_b_impl);

  // Trigger the instantiation of a template in 'a' that uses a type defined in
  // 'b_impl'. That type is not visible here, nor in 'a'. This fails; there is
  // no reason why DefinedInBImpl should be visible here.
  //
  // We turn off error recovery for modules in this test (so we don't get an
  // implicit import of cxx_templates_b_impl), and that results in us producing
  // a big spew of errors here.
  //
  // expected-error@Inputs/cxx-templates-a.h:19 {{definition of 'DefinedInBImpl' must be imported}}
  // expected-note@Inputs/cxx-templates-b-impl.h:1 +{{definition here is not reachable}}
  // expected-error@Inputs/cxx-templates-a.h:19 +{{}}
  // expected-error@Inputs/cxx-templates-a.h:20 +{{}}
  PerformDelayedLookup(defined_in_b_impl); // expected-note {{in instantiation of}}

  merge_templates_a = merge_templates_b; // ok, same type

  using T = decltype(enum_a_from_a);
  using T = decltype(enum_b_from_b);
  T e = true ? enum_a_from_a : enum_b_from_b;

  UseRedeclaredEnum<int>(UseInt<1>());
  // FIXME: Reintroduce this once we merge function template specializations.
  //static_assert(UseRedeclaredEnumA == UseRedeclaredEnumB, "");
  //static_assert(UseRedeclaredEnumA == UseRedeclaredEnum<int>, "");
  //static_assert(UseRedeclaredEnumB == UseRedeclaredEnum<int>, "");
  static_assert(enum_c_from_a == enum_c_from_b, "");
  CommonTemplate<int> cti;
  CommonTemplate<int>::E eee = CommonTemplate<int>::c;

  TemplateInstantiationVisibility<char[1]> tiv1;
  TemplateInstantiationVisibility<char[2]> tiv2;
  TemplateInstantiationVisibility<char[3]> tiv3; // expected-error 5{{must be imported from module 'cxx_templates_b_impl'}}
  // expected-note@cxx-templates-b-impl.h:10 3{{explicit specialization declared here is not reachable}}
  // expected-note@cxx-templates-b-impl.h:10 2{{definition here is not reachable}}
  TemplateInstantiationVisibility<char[4]> tiv4;

  int &p = WithPartialSpecializationUse().f();
  int &q = WithExplicitSpecializationUse().inner_template<int>();
  int *r = PartiallyInstantiatePartialSpec<int*>::bar();

  (void)&WithImplicitSpecialMembers<int>::n;

  MergeClassTemplateSpecializations_string s;

  extern TestInjectedClassName::A *use_a;
  extern TestInjectedClassName::C *use_c;
  TestInjectedClassName::UseD();
}

static_assert(Outer<int>::Inner<int>::f() == 1, "");
static_assert(Outer<int>::Inner<int>::g() == 2, "");

static_assert(MergeTemplateDefinitions<int>::f() == 1, "");
static_assert(MergeTemplateDefinitions<int>::g() == 2, "");

RedeclaredAsFriend<int> raf1;
RedeclareTemplateAsFriend<double> rtaf;
RedeclaredAsFriend<double> raf2;

MergeSpecializations<int*>::partially_specialized_in_a spec_in_a_1;
MergeSpecializations<int&>::partially_specialized_in_b spec_in_b_1;
MergeSpecializations<int[]>::partially_specialized_in_c spec_in_c_1;
MergeSpecializations<char>::explicitly_specialized_in_a spec_in_a_2;
MergeSpecializations<double>::explicitly_specialized_in_b spec_in_b_2;
MergeSpecializations<bool>::explicitly_specialized_in_c spec_in_c_2;

MergeAnonUnionMember<> maum_main;
typedef DontWalkPreviousDeclAfterMerging<int> dwpdam_typedef_2;
dwpdam_typedef::type dwpdam_typedef_use;
DontWalkPreviousDeclAfterMerging<int>::Inner::type dwpdam;

using AliasTemplateMergingTest = WithAliasTemplate<int>::X<char>;

int AnonymousDeclsMergingTest(WithAnonymousDecls<int> WAD, WithAnonymousDecls<char> WADC) {
  return InstantiateWithAnonymousDeclsA(WAD) +
         InstantiateWithAnonymousDeclsB(WAD) +
         InstantiateWithAnonymousDeclsB2(WADC) +
         InstantiateWithAnonymousDeclsD(WADC);
}

@import cxx_templates_common;

typedef SomeTemplate<int*> SomeTemplateIntPtr;
typedef SomeTemplate<int&> SomeTemplateIntRef;
SomeTemplate<char*> some_template_char_ptr;
SomeTemplate<char&> some_template_char_ref;

void testImplicitSpecialMembers(SomeTemplate<char[1]> &a,
                                const SomeTemplate<char[1]> &b,
                                SomeTemplate<char[2]> &c,
                                const SomeTemplate<char[2]> &d) {
  a = b;
  c = d;
}

bool testFriendInClassTemplate(Std::WithFriend<int> wfi) {
  return wfi != wfi;
}

namespace hidden_specializations {
  // expected-note@cxx-templates-unimported.h:* 1+{{here}}
  void test() {
    // For functions, uses that would trigger instantiations of definitions are
    // not allowed.
    fn<void>(); // ok
    fn<char>(); // ok
    fn<int>(); // expected-error 1+{{explicit specialization of 'fn<int>' must be imported}}
    cls<void>::nested_fn(); // expected-error 1+{{explicit specialization of 'nested_fn' must be imported}}
    cls<void>::nested_fn_t<int>(); // expected-error 1+{{explicit specialization of 'nested_fn_t' must be imported}}
    cls<void>::nested_fn_t<char>(); // expected-error 1+{{explicit specialization of 'nested_fn_t' must be imported}}

    // For classes, uses that would trigger instantiations of definitions are
    // not allowed.
    cls<void> *k0; // ok
    cls<char> *k1; // ok
    cls<int> *k2; // ok
    cls<int*> *k3; // ok
    cls<void>::nested_cls *nk1; // ok
    cls<void>::nested_cls_t<int> *nk2; // ok
    cls<void>::nested_cls_t<char> *nk3; // ok
    cls<int> uk1; // expected-error 1+{{explicit specialization of 'cls<int>' must be imported}} expected-error 1+{{definition of}}
    cls<int*> uk3; // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}} expected-error 1+{{definition of}}
    cls<char*> uk4; // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}} expected-error 1+{{definition of}}
    cls<void>::nested_cls unk1; // expected-error 1+{{explicit specialization of 'nested_cls' must be imported}} expected-error 1+{{definition of}}
    cls<void>::nested_cls_t<int> unk2; // expected-error 1+{{explicit specialization of 'nested_cls_t' must be imported}} expected-error 1+{{definition of}}
    // expected-error@cxx-templates-unimported.h:29 {{explicit specialization of 'nested_cls_t' must be imported}}
    // expected-note@-2 {{in evaluation of exception specification}}
    cls<void>::nested_cls_t<char> unk3; // expected-error 1+{{explicit specialization of 'nested_cls_t' must be imported}}

    // For enums, uses that would trigger instantiations of definitions are not
    // allowed.
    cls<void>::nested_enum e; // ok
    (void)cls<void>::nested_enum::e; // expected-error 1+{{definition of 'nested_enum' must be imported}} expected-error 1+{{declaration of 'e'}}

    // For variable template specializations, no uses are allowed because
    // specializations can change the type.
    (void)sizeof(var<void>); // ok
    (void)sizeof(var<char>); // ok
    (void)sizeof(var<int>); // expected-error 1+{{explicit specialization of 'var<int>' must be imported}}
    (void)sizeof(var<int*>); // expected-error 1+{{partial specialization of 'var<T *>' must be imported}}
    (void)sizeof(var<char*>); // expected-error 1+{{partial specialization of 'var<T *>' must be imported}}
    (void)sizeof(cls<void>::nested_var); // ok
    (void)cls<void>::nested_var; // expected-error 1+{{explicit specialization of 'nested_var' must be imported}}
    (void)sizeof(cls<void>::nested_var_t<int>); // expected-error 1+{{explicit specialization of 'nested_var_t' must be imported}}
    (void)sizeof(cls<void>::nested_var_t<char>); // expected-error 1+{{explicit specialization of 'nested_var_t' must be imported}}
  }

  void cls<int>::nested_fn() {} // expected-error 1+{{explicit specialization of 'cls<int>' must be imported}} expected-error 1+{{definition of}}
  struct cls<int>::nested_cls {}; // expected-error 1+{{explicit specialization of 'cls<int>' must be imported}} expected-error 1+{{definition of}}
  int cls<int>::nested_var; // expected-error 1+{{explicit specialization of 'cls<int>' must be imported}} expected-error 1+{{definition of}}
  enum cls<int>::nested_enum : int {}; // expected-error 1+{{explicit specialization of 'cls<int>' must be imported}} expected-error 1+{{definition of}}

  template<typename T> void cls<T*>::nested_fn() {} // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}}
  template<typename T> struct cls<T*>::nested_cls {}; // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}}
  template<typename T> int cls<T*>::nested_var; // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}}
  template<typename T> enum cls<T*>::nested_enum : int {}; // expected-error 1+{{partial specialization of 'cls<T *>' must be imported}}
}

namespace Std {
  void g(); // expected-error {{functions that differ only in their return type cannot be overloaded}}
  // expected-note@cxx-templates-common.h:21 {{previous}}
}

// CHECK-GLOBAL:      DeclarationName 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: `-FunctionTemplate {{.*}} 'f'

// CHECK-NAMESPACE-N:      DeclarationName 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: `-FunctionTemplate {{.*}} 'f'

// CHECK-DUMP:      ClassTemplateDecl {{.*}} <{{.*[/\\]}}cxx-templates-common.h:1:1, {{.*}}>  col:{{.*}} in cxx_templates_common SomeTemplate
// CHECK-DUMP:        ClassTemplateSpecializationDecl {{.*}} prev {{.*}} SomeTemplate
// CHECK-DUMP-NEXT:     TemplateArgument type 'char[2]'
// CHECK-DUMP:        ClassTemplateSpecializationDecl {{.*}} SomeTemplate definition
// CHECK-DUMP-NEXT:     DefinitionData
// CHECK-DUMP-NEXT:       DefaultConstructor
// CHECK-DUMP-NEXT:       CopyConstructor
// CHECK-DUMP-NEXT:       MoveConstructor
// CHECK-DUMP-NEXT:       CopyAssignment
// CHECK-DUMP-NEXT:       MoveAssignment
// CHECK-DUMP-NEXT:       Destructor
// CHECK-DUMP-NEXT:     TemplateArgument type 'char[2]'
// CHECK-DUMP:        ClassTemplateSpecializationDecl {{.*}} prev {{.*}} SomeTemplate
// CHECK-DUMP-NEXT:     TemplateArgument type 'char[1]'
// CHECK-DUMP:        ClassTemplateSpecializationDecl {{.*}} SomeTemplate definition
// CHECK-DUMP-NEXT:     DefinitionData
// CHECK-DUMP-NEXT:       DefaultConstructor
// CHECK-DUMP-NEXT:       CopyConstructor
// CHECK-DUMP-NEXT:       MoveConstructor
// CHECK-DUMP-NEXT:       CopyAssignment
// CHECK-DUMP-NEXT:       MoveAssignment
// CHECK-DUMP-NEXT:       Destructor
// CHECK-DUMP-NEXT:     TemplateArgument type 'char[1]'
