// RUN: %check_clang_tidy %s bugprone-unused-return-value %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: bugprone-unused-return-value.CheckedFunctions, \
// RUN:    value: "::fun;::ns::Outer::Inner::memFun;::ns::Type::staticFun;::ns::ClassTemplate::memFun;::ns::ClassTemplate::staticFun"}]}' \
// RUN: --

namespace std {

template <typename T>
T *launder(T *);

} // namespace std

namespace ns {

struct Outer {
  struct Inner {
    bool memFun();
  };
};

using AliasName = Outer;

struct Derived : public Outer::Inner {};

struct Retval {
  int *P;
  Retval() { P = new int; }
  ~Retval() { delete P; }
};

struct Type {
  Retval memFun();
  static Retval staticFun();
};

template <typename T>
struct ClassTemplate {
  Retval memFun();
  static Retval staticFun();
};

} // namespace ns

int fun();
void fun(int);

void warning() {
  fun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value returned by this function should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  (fun());
  // CHECK-NOTES: [[@LINE-1]]:4: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:4: note: cast {{.*}} this warning

  ns::Outer::Inner ObjA1;
  ObjA1.memFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning

  ns::AliasName::Inner ObjA2;
  ObjA2.memFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning

  ns::Derived ObjA3;
  ObjA3.memFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning

  ns::Type::staticFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning

  ns::ClassTemplate<int> ObjA4;
  ObjA4.memFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning

  ns::ClassTemplate<int>::staticFun();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: the value {{.*}} should be used
  // CHECK-NOTES: [[@LINE-2]]:3: note: cast {{.*}} this warning
}

void noWarning() {
  auto R1 = fun();

  ns::Outer::Inner ObjB1;
  auto R2 = ObjB1.memFun();

  auto R3 = ns::Type::staticFun();

  ns::ClassTemplate<int> ObjB2;
  auto R4 = ObjB2.memFun();

  auto R5 = ns::ClassTemplate<int>::staticFun();

  // test calling a void overload of a checked function
  fun(5);

  // test discarding return value of functions that are not configured to be checked
  int I = 1;
  std::launder(&I);

  ns::Type ObjB3;
  ObjB3.memFun();
}
