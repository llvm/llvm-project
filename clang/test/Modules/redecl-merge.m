// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -Wreturn-type -fmodules-cache-path=%t -I %S/Inputs %s -verify -Wno-objc-root-class

@class C2;
@class C3;
@class C3;
@import redecl_merge_left;
typedef struct my_struct_type *my_struct_ref;
@protocol P4;
@class C3;
@class C3;

int *call_eventually_noreturn(void) {
  eventually_noreturn();
} // expected-warning{{non-void function does not return a value}}

int *call_eventually_noreturn2(void) {
  eventually_noreturn2();
} // expected-warning{{non-void function does not return a value}}

@import redecl_merge_right;

int *call_eventually_noreturn_again(void) {
  eventually_noreturn();
}

int *call_eventually_noreturn2_again(void) {
  // noreturn and non-noreturn functions have different types
  eventually_noreturn2(); // expected-error{{call to 'eventually_noreturn2' is ambiguous}}
  // expected-note@Inputs/redecl-merge-left.h:93{{candidate function}}
  // expected-note@Inputs/redecl-merge-right.h:90{{candidate function}}
}

@implementation A
- (Super*)init { return self; }
@end

void f(A *a) {
  [a init];
}

@class A;

B *f1() {
  return [B create_a_B];
}

@class B;

void testProtoMerge(id<P1> p1, id<P2> p2) {
  [p1 protoMethod1];
  [p2 protoMethod2];
}

struct S1 {
  int s1_field;
};

struct S3 {
  int s3_field;
};

void testTagMerge() {
  consume_S1(produce_S1());
  struct S2 s2;
  s2.field = 0;
  consume_S2(produce_S2());
  struct S1 s1;
  s1.s1_field = 0;
  consume_S3(produce_S3());
  struct S4 s4;
  s4.field = 0;
  consume_S4(produce_S4());
  struct S3 s3;
  s3.s3_field = 0;
}

void testTypedefMerge(int i, double d) {
  T1 *ip = &i;
  // FIXME: Typedefs aren't actually merged in the sense of other merges, because
  // we should only merge them when the types are identical.
  // expected-note@Inputs/redecl-merge-left.h:60{{candidate found by name lookup is 'T2'}}
  // expected-note@Inputs/redecl-merge-right.h:63{{candidate found by name lookup is 'T2'}}
  T2 *dp = &d; // expected-error{{reference to 'T2' is ambiguous}}
}

void testFuncMerge(int i) {
  func0(i);
  func1(i);
  // expected-note@Inputs/redecl-merge-left.h:64{{candidate function}}
  // expected-note@Inputs/redecl-merge-right.h:70{{candidate function}}
  func2(i); // expected-error{{call to 'func2' is ambiguous}}
}

void testVarMerge(int i) {
  var1 = i;
  // expected-note@Inputs/redecl-merge-left.h:77{{candidate found by name lookup is 'var2'}}
  // expected-note@Inputs/redecl-merge-right.h:77{{candidate found by name lookup is 'var2'}}
  var2 = i; // expected-error{{reference to 'var2' is ambiguous}}
  // expected-note@Inputs/redecl-merge-left.h:79{{candidate found by name lookup is 'var3'}}
  // expected-note@Inputs/redecl-merge-right.h:79{{candidate found by name lookup is 'var3'}}
  var3 = i; // expected-error{{reference to 'var3' is ambiguous}}
}

// Test redeclarations of entities in explicit submodules, to make
// sure we're maintaining the declaration chains even when normal name
// lookup can't see what we're looking for.
void testExplicit() {
  Explicit *e;
  int *(*fp)(void) = &explicit_func;
  int *ip = explicit_func();

  // FIXME: Should complain about definition not having been imported.
  struct explicit_struct es = { 0 };
}

// Test resolution of declarations from multiple modules with no
// common original declaration.
void test_C(C *c) {
  c = get_a_C();
  accept_a_C(c);
}

void test_C2(C2 *c2) {
  c2 = get_a_C2();
  accept_a_C2(c2);
}

void test_C3(C3 *c3) {
  c3 = get_a_C3();
  accept_a_C3(c3);
}

C4 *global_C4;

ClassWithDef *cwd1;

@import redecl_merge_left_left;

void test_C4a(C4 *c4) {
  global_C4 = c4 = get_a_C4();
  accept_a_C4(c4);
}

void test_ClassWithDef(ClassWithDef *cwd) {
  [cwd method];
}

@import redecl_merge_bottom;

void test_C4b() {
  if (&refers_to_C4) {
  }
}

@implementation B
+ (B*)create_a_B { return 0; }
@end

void g(A *a) {
  [a init];
}

@protocol P3
- (void)p3_method;
@end

id<P4> p4;
id<P3> p3;

// Make sure we don't get conflicts with 'id'.
funcptr_with_id fid;
id id_global;


