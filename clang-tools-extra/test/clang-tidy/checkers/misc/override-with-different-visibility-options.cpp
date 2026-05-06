// RUN: %check_clang_tidy -check-suffixes=DTORS,WIDENING,NARROWING %s misc-override-with-different-visibility %t -- \
// RUN:   -config="{CheckOptions: {misc-override-with-different-visibility.CheckDestructors: true}}"

// RUN: %check_clang_tidy -check-suffixes=OPS,WIDENING,NARROWING %s misc-override-with-different-visibility %t -- \
// RUN:   -config="{CheckOptions: {misc-override-with-different-visibility.CheckOperators: true}}"

// RUN: %check_clang_tidy -check-suffixes=WIDENING %s misc-override-with-different-visibility %t -- \
// RUN:   -config="{CheckOptions: {misc-override-with-different-visibility.DisallowedVisibilityChange: 'widening'}}"

// RUN: %check_clang_tidy -check-suffixes=NARROWING %s misc-override-with-different-visibility %t -- \
// RUN:   -config="{CheckOptions: {misc-override-with-different-visibility.DisallowedVisibilityChange: 'narrowing'}}"

namespace test_change {

class A {
protected:
  virtual void f1();
  virtual void f2();
};

class B: public A {
public:
  void f1();
  // CHECK-MESSAGES-WIDENING: :[[@LINE-1]]:8: warning: visibility of function 'f1'
  // CHECK-MESSAGES-WIDENING: :[[@LINE-8]]:16: note: function declared here
private:
  void f2();
  // CHECK-MESSAGES-NARROWING: :[[@LINE-1]]:8: warning: visibility of function 'f2'
  // CHECK-MESSAGES-NARROWING: :[[@LINE-11]]:16: note: function declared here
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
  // CHECK-MESSAGES-DTORS: :[[@LINE-1]]:3: warning: visibility of function '~B'
  // CHECK-MESSAGES-DTORS: :[[@LINE-7]]:11: note: function declared here
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
  // CHECK-MESSAGES-OPS: :[[@LINE-1]]:6: warning: visibility of function 'operator='
  // CHECK-MESSAGES-OPS: :[[@LINE-10]]:14: note: function declared here
  A& operator++();
  // CHECK-MESSAGES-OPS: :[[@LINE-1]]:6: warning: visibility of function 'operator++'
  // CHECK-MESSAGES-OPS: :[[@LINE-12]]:14: note: function declared here
  int operator()(int);
  // CHECK-MESSAGES-OPS: :[[@LINE-1]]:7: warning: visibility of function 'operator()'
  // CHECK-MESSAGES-OPS: :[[@LINE-14]]:15: note: function declared here
  operator double() const;
  // CHECK-MESSAGES-OPS: :[[@LINE-1]]:3: warning: visibility of function 'operator double'
  // CHECK-MESSAGES-OPS: :[[@LINE-16]]:11: note: function declared here
};

}
