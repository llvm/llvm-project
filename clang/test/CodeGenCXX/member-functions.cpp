// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-apple-darwin9 -o %t
struct C {
  void f();
  void g(int, ...);
};

// RUN: grep "define void @_ZN1C1fEv" %t | count 1
void C::f() {
}

void test1() {
  C c;
  
// RUN: grep "call void @_ZN1C1fEv" %t | count 1
  c.f();
  
// RUN: grep "call void (.struct.C\*, i32, ...)\* @_ZN1C1gEiz" %t | count 1
  c.g(1, 2, 3);
}


struct S {
  // RUN: grep "define linkonce_odr void @_ZN1SC1Ev.*unnamed_addr" %t
  inline S() { }
  // RUN: grep "define linkonce_odr void @_ZN1SC1Ev.*unnamed_addr" %t
  inline ~S() { }
  
  
  // RUN: grep "define linkonce_odr void @_ZN1S9f_inline1Ev" %t
  void f_inline1() { }
  // RUN: grep "define linkonce_odr void @_ZN1S9f_inline2Ev" %t
  inline void f_inline2() { }
  
  // RUN: grep "define linkonce_odr void @_ZN1S1gEv" %t
  static void g() { }
  
  static void f();

  // RUN: grep "define linkonce_odr void @_ZN1S1vEv.*unnamed_addr" %t
  virtual void v() {}
};

// RUN: grep "define void @_ZN1S1fEv" %t
void S::f() {
}

void test2() {
  S s;
  
  s.f_inline1();
  s.f_inline2();
  
  S::g();
  
}

struct T {
  T operator+(const T&);
};

void test3() {
  T t1, t2;
  
  // RUN: grep "call void @_ZN1TplERKS_" %t
  T result = t1 + t2;
}
