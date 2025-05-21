// RUN: %check_clang_tidy -check-suffixes=DTORS,WIDENING,NARROWING %s bugprone-visibility-change-to-virtual-function %t -- \
// RUN:   -config="{CheckOptions: {bugprone-visibility-change-to-virtual-function.CheckDestructors: true}}"

// RUN: %check_clang_tidy -check-suffixes=OPS,WIDENING,NARROWING %s bugprone-visibility-change-to-virtual-function %t -- \
// RUN:   -config="{CheckOptions: {bugprone-visibility-change-to-virtual-function.CheckOperators: true}}"

// RUN: %check_clang_tidy -check-suffixes=WIDENING %s bugprone-visibility-change-to-virtual-function %t -- \
// RUN:   -config="{CheckOptions: {bugprone-visibility-change-to-virtual-function.DisallowedVisibilityChange: 'widening'}}"

// RUN: %check_clang_tidy -check-suffixes=NARROWING %s bugprone-visibility-change-to-virtual-function %t -- \
// RUN:   -config="{CheckOptions: {bugprone-visibility-change-to-virtual-function.DisallowedVisibilityChange: 'narrowing'}}"

namespace test_change {

class A {
protected:
  virtual void f1();
  virtual void f2();
};

class B: public A {
public:
  void f1();
  // CHECK-NOTES-WIDENING: :[[@LINE-1]]:8: warning: visibility of function 'f1'
  // CHECK-NOTES-WIDENING: :[[@LINE-8]]:16: note: function declared here
private:
  void f2();
  // CHECK-NOTES-NARROWING: :[[@LINE-1]]:8: warning: visibility of function 'f2'
  // CHECK-NOTES-NARROWING: :[[@LINE-11]]:16: note: function declared here
};

}

namespace test_destructor {

class A {
public:
  virtual ~A();
};

class B: public A {
protected:
  ~B();
  // CHECK-NOTES-DTORS: :[[@LINE-1]]:3: warning: visibility of function '~B'
  // CHECK-NOTES-DTORS: :[[@LINE-7]]:11: note: function declared here
};

}

namespace test_operator {

class A {
  virtual A& operator=(const A&);
  virtual A& operator++();
  virtual int operator()(int);
  virtual operator double() const;
};

class B: public A {
protected:
  A& operator=(const A&);
  // CHECK-NOTES-OPS: :[[@LINE-1]]:6: warning: visibility of function 'operator='
  // CHECK-NOTES-OPS: :[[@LINE-10]]:14: note: function declared here
  A& operator++();
  // CHECK-NOTES-OPS: :[[@LINE-1]]:6: warning: visibility of function 'operator++'
  // CHECK-NOTES-OPS: :[[@LINE-12]]:14: note: function declared here
  int operator()(int);
  // CHECK-NOTES-OPS: :[[@LINE-1]]:7: warning: visibility of function 'operator()'
  // CHECK-NOTES-OPS: :[[@LINE-14]]:15: note: function declared here
  operator double() const;
  // CHECK-NOTES-OPS: :[[@LINE-1]]:3: warning: visibility of function 'operator double'
  // CHECK-NOTES-OPS: :[[@LINE-16]]:11: note: function declared here
};

}
