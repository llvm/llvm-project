// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -verify %s -Wno-objc-root-class
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -emit-llvm %s -o - -Wno-objc-root-class | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
@import templates_left;

void testInlineRedeclEarly() {
  // instantiate definition now, we'll add another declaration in _right.
  OutOfLineInline<int>().h();
}

@import templates_right;

// CHECK-DAG: @list_left = global %[[LIST:.*]] { %[[LISTNODE:.*]]* null, i32 8 }, align 8
// CHECK-DAG: @list_right = global %[[LIST]] { %[[LISTNODE]]* null, i32 12 }, align 8
// CHECK-DAG: @__const._Z15testMixedStructv.l = {{.*}} constant %[[LIST]] { %{{.*}}* null, i32 1 }, align 8
// CHECK-DAG: @__const._Z15testMixedStructv.r = {{.*}} constant %[[LIST]] { %{{.*}}* null, i32 2 }, align 8
// CHECK-DAG: @_ZN29WithUndefinedStaticDataMemberIA_iE9undefinedE = external global

void testTemplateClasses() {
  Vector<int> vec_int;
  vec_int.push_back(0);

  List<bool> list_bool;
  list_bool.push_back(false);

  N::Set<char> set_char;
  set_char.insert('A');

  static_assert(sizeof(List<long>) == sizeof(List<short>), "");

  List<double> list_double;
  list_double.push_back(0.0);
}

void testPendingInstantiations() {
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]i}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]double}}
  triggerPendingInstantiation();
  triggerPendingInstantiationToo();
}

void testRedeclDefinition() {
  // CHECK: define {{.*redeclDefinitionEmit}}
  redeclDefinitionEmit();
}

void testInlineRedecl() {
  outOfLineInlineUseLeftF();
  outOfLineInlineUseRightG();

  outOfLineInlineUseRightF();
  outOfLineInlineUseLeftG();
}

// CHECK-NOT: @_ZN21ExplicitInstantiationILb0ELb0EE1fEv(
// CHECK: declare {{.*}}@_ZN21ExplicitInstantiationILb1ELb0EE1fEv(
// CHECK: define {{.*}}@_ZN21ExplicitInstantiationILb1ELb1EE1fEv(
// CHECK-NOT: @_ZN21ExplicitInstantiationILb0ELb0EE1fEv(

// These three are all the same type.
typedef OuterIntInner_left OuterIntInner;
typedef OuterIntInner_right OuterIntInner;
typedef Outer<int>::Inner OuterIntInner;

// CHECK: call {{.*pendingInstantiation}}
// CHECK: call {{.*redeclDefinitionEmit}}

static_assert(size_left == size_right, "same field both ways");
void useListInt(List<int> &);

// CHECK-LABEL: define i32 @_Z15testMixedStructv(
unsigned testMixedStruct() {
  // CHECK: %[[l:.*]] = alloca %[[ListInt:[^ ]*]], align 8
  // CHECK: %[[r:.*]] = alloca %[[ListInt]], align 8

  // CHECK: call {{.*}}memcpy{{.*}}(i8* align {{[0-9]+}} %{{.*}}, i8* align {{[0-9]+}} bitcast ({{.*}}* @__const._Z15testMixedStructv.l to i8*), i64 16,
  ListInt_left l{0, 1};

  // CHECK: call {{.*}}memcpy{{.*}}(i8* align {{[0-9]+}} %{{.*}}, i8* align {{[0-9]+}} bitcast ({{.*}}* @__const._Z15testMixedStructv.r to i8*), i64 16,
  ListInt_right r{0, 2};

  // CHECK: call void @_Z10useListIntR4ListIiE(%[[ListInt]]* dereferenceable({{[0-9]+}}) %[[l]])
  useListInt(l);
  // CHECK: call void @_Z10useListIntR4ListIiE(%[[ListInt]]* dereferenceable({{[0-9]+}}) %[[r]])
  useListInt(r);

  // CHECK: load i32, i32* bitcast (i8* getelementptr inbounds (i8, i8* bitcast ({{.*}}* @list_left to i8*), i64 8) to i32*)
  // CHECK: load i32, i32* bitcast (i8* getelementptr inbounds (i8, i8* bitcast ({{.*}}* @list_right to i8*), i64 8) to i32*)
  return list_left.*size_right + list_right.*size_left;
}

template<typename T> struct MergePatternDecl {
  typedef int Type;
  void f(Type);
};
template<typename T> void MergePatternDecl<T>::f(Type type) {}
// CHECK: define {{.*}}@_ZN21ExplicitInstantiationILb0ELb1EE1fEv(
template struct ExplicitInstantiation<false, true>;
template struct ExplicitInstantiation<true, true>;

void testDelayUpdatesImpl() { testDelayUpdates<int>(); }

void testStaticDataMember() {
  WithUndefinedStaticDataMember<int[]> load_it;

  // CHECK-LABEL: define linkonce_odr i32* @_Z23getStaticDataMemberLeftv(
  // CHECK: ret i32* getelementptr inbounds ([0 x i32], [0 x i32]* @_ZN29WithUndefinedStaticDataMemberIA_iE9undefinedE, i32 0, i32 0)
  (void) getStaticDataMemberLeft();

  // CHECK-LABEL: define linkonce_odr i32* @_Z24getStaticDataMemberRightv(
  // CHECK: ret i32* getelementptr inbounds ([0 x i32], [0 x i32]* @_ZN29WithUndefinedStaticDataMemberIA_iE9undefinedE, i32 0, i32 0)
  (void) getStaticDataMemberRight();
}

void testWithAttributes() {
  auto a = make_with_attributes_left();
  auto b = make_with_attributes_right();
  static_assert(alignof(decltype(a)) == 2, "");
  static_assert(alignof(decltype(b)) == 2, "");
}

// Check that returnNonTrivial doesn't return Class0<S0> directly in registers.

// CHECK: declare void @_Z16returnNonTrivialv(%struct.Class0* sret)

@import template_nontrivial0;
@import template_nontrivial1;

S1::S1() : a(returnNonTrivial()) {
}
