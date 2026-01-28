// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++23 %s -emit-llvm -o - | FileCheck %s

struct Base {

  void no_args_1(void (*callback)(void));
  __attribute__((callback(1))) void no_args_2(void (*callback1)(void), void (*callback2)(void));
  __attribute__((callback(callback1))) void no_args_3(void (*callback1)(void), void (*callback2)(void));

  // TODO: There should probably be a warning or even an error for different
  //       callbacks on the same method.
  __attribute__((callback(1))) virtual void
  virtual_1(void (*callback)(void));

  __attribute__((callback(callback, this, __, this))) virtual void
  this_unknown_this(void (*callback)(Base *, Base *, Base *));
};

// CHECK-DAG:      define{{.*}} void @_ZN4Base9no_args_1EPFvvE({{[^!]*!callback}} ![[cid0:[0-9]+]]
__attribute__((callback(1))) void
Base::no_args_1(void (*callback)(void)) {
}

// CHECK-DAG:      define{{.*}} void @_ZN4Base9no_args_2EPFvvES1_({{[^!]*!callback}} ![[cid1:[0-9]+]]
__attribute__((callback(2))) void Base::no_args_2(void (*callback1)(void), void (*callback2)(void)) {
}
// CHECK-DAG:      define{{.*}} void @_ZN4Base9no_args_3EPFvvES1_({{[^!]*!callback}} ![[cid1]]
__attribute__((callback(callback2))) void Base::no_args_3(void (*callback1)(void), void (*callback2)(void)) {
}

// CHECK-DAG:      define{{.*}} void @_ZN4Base17this_unknown_thisEPFvPS_S0_S0_E({{[^!]*!callback}} ![[cid2:[0-9]+]]
void Base::this_unknown_this(void (*callback)(Base *, Base *, Base *)) {
}

struct Derived_1 : public Base {
  __attribute__((callback(1))) virtual void
  virtual_1(void (*callback)(void)) override;
};

// CHECK-DAG:      define{{.*}} void @_ZN9Derived_19virtual_1EPFvvE({{[^!]*!callback}} ![[cid0]]
void Derived_1::virtual_1(void (*callback)(void)) {}

struct Derived_2 : public Base {
  void virtual_1(void (*callback)(void)) override;
};

// CHECK-DAG: define{{.*}} void @_ZN9Derived_29virtual_1EPFvvE
// CHECK-NOT: !callback
void Derived_2::virtual_1(void (*callback)(void)) {}

class ExplicitParameterObject {
  __attribute__((callback(1, 0))) void implicit_this_idx(void (*callback)(ExplicitParameterObject*));
  __attribute__((callback(1, this))) void implicit_this_identifier(void (*callback)(ExplicitParameterObject*));
  __attribute__((callback(2, 1))) void explicit_this_idx(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*));
  __attribute__((callback(2, self))) void explicit_this_identifier(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*));
};

// CHECK-DAG: define{{.*}} void @_ZN23ExplicitParameterObject17implicit_this_idxEPFvPS_E({{[^!]*!callback}} ![[cid3:[0-9]+]]
void ExplicitParameterObject::implicit_this_idx(void (*callback)(ExplicitParameterObject*)) {}

// CHECK-DAG: define{{.*}} void @_ZN23ExplicitParameterObject24implicit_this_identifierEPFvPS_E({{[^!]*!callback}} ![[cid3]]
void ExplicitParameterObject::implicit_this_identifier(void (*callback)(ExplicitParameterObject*)) {}

// CHECK-DAG: define{{.*}} void @_ZNH23ExplicitParameterObject17explicit_this_idxEPS_PFvS0_E({{[^!]*!callback}} ![[cid3]]
void ExplicitParameterObject::explicit_this_idx(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*)) {}

// CHECK-DAG: define{{.*}} void @_ZNH23ExplicitParameterObject24explicit_this_identifierEPS_PFvS0_E({{[^!]*!callback}} ![[cid3]]
void ExplicitParameterObject::explicit_this_identifier(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*)) {}

// CHECK-DAG: ![[cid0]] = !{![[cid0b:[0-9]+]]}
// CHECK-DAG: ![[cid0b]] = !{i64 1, i1 false}
// CHECK-DAG: ![[cid1]] = !{![[cid1b:[0-9]+]]}
// CHECK-DAG: ![[cid1b]] = !{i64 2, i1 false}
// CHECK-DAG: ![[cid2]] = !{![[cid2b:[0-9]+]]}
// CHECK-DAG: ![[cid2b]] = !{i64 1, i64 0, i64 -1, i64 0, i1 false}
// CHECK-DAG: ![[cid3]] = !{![[cid3b:[0-9]+]]}
// CHECK-DAG: ![[cid3b]] = !{i64 1, i64 0, i1 false}
