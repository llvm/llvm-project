// RUN: %check_clang_tidy %s misc-override-with-different-visibility %t -- \
// RUN:   -config="{CheckOptions: {misc-override-with-different-visibility.IgnoredFunctions: 'IgnoreAlways::.*;::a::IgnoreSelected::.*;IgnoreFunctions::f1;ignored_f'}}"

class IgnoreAlways {
  virtual void f();
};

class IgnoreSelected {
  virtual void f();
};

namespace a {
class IgnoreAlways {
  virtual void f();
};
class IgnoreSelected {
  virtual void f();
};
}

namespace ignore_always {
class Test1: public IgnoreAlways {
public:
  void f();
  void ignored_f(int);
};
class Test2: public a::IgnoreAlways {
public:
  void f();
};
}

namespace ignore_selected {
class Test1: public IgnoreSelected {
public:
  void f();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'f'
  // CHECK-MESSAGES: :9:16: note: function declared here
  void ignored_f(int);
};
class Test2: public a::IgnoreSelected {
public:
  void f();
};
}

class IgnoreFunctions {
  virtual void f1();
  virtual void f2();
  virtual void ignored_f();
};

class IgnoreFunctionsTest: public IgnoreFunctions {
public:
  void f1();
  void f2();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'f2'
  // CHECK-MESSAGES: :[[@LINE-9]]:16: note: function declared here
  void ignored_f();
};
