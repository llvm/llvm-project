// RUN: %check_clang_tidy %s bugprone-unused-return-value %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: bugprone-unused-return-value.CheckedFunctions, \
// RUN:    value: "::fun;::ns::Outer::Inner::memFun;::ns::Type::staticFun"}]}' \
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

} // namespace ns

int fun();
void fun(int);

void warning() {
  fun();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should be used [bugprone-unused-return-value]

  (fun());
  // CHECK-MESSAGES: [[@LINE-1]]:4: warning: the value returned by this function should be used [bugprone-unused-return-value]

  ns::Outer::Inner ObjA1;
  ObjA1.memFun();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should be used [bugprone-unused-return-value]

  ns::AliasName::Inner ObjA2;
  ObjA2.memFun();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should be used [bugprone-unused-return-value]

  ns::Derived ObjA3;
  ObjA3.memFun();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should be used [bugprone-unused-return-value]

  ns::Type::staticFun();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should be used [bugprone-unused-return-value]
}

void noWarning() {
  auto R1 = fun();

  ns::Outer::Inner ObjB1;
  auto R2 = ObjB1.memFun();

  auto R3 = ns::Type::staticFun();

  // test calling a void overload of a checked function
  fun(5);

  // test discarding return value of functions that are not configured to be checked
  int I = 1;
  std::launder(&I);

  ns::Type ObjB2;
  ObjB2.memFun();
}
