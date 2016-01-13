// RUN: %check_clang_tidy %s misc-virtual-near-miss %t

struct Base {
  virtual void func();
  virtual void gunk();
};

struct Derived : Base {
  // Should not warn "do you want to override 'gunk'?", because gunk is already
  // overriden by this class.
  virtual void funk();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::funk' has a similar name and the same signature as virtual method 'Base::func'; did you mean to override it? [misc-virtual-near-miss]

  void func2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::func2' has {{.*}} 'Base::func'

  void func22(); // Should not warn.

  void gunk(); // Should not warn: gunk is override.

  void fun();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::fun' has {{.*}} 'Base::func'
};

class Father {
public:
  Father();
  virtual void func();
  virtual Father *create(int i);
  virtual Base &&generate();
};

class Mother {
public:
  Mother();
  static void method();
  virtual int method(int argc, const char **argv);
  virtual int method(int argc) const;
};

class Child : private Father, private Mother {
public:
  Child();

  virtual void func2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::func2' has {{.*}} 'Father::func'

  int methoe(int x, char **strs); // Should not warn: parameter types don't match.

  int methoe(int x); // Should not warn: method is not const.

  void methof(int x, const char **strs); // Should not warn: return types don't match.

  int methoh(int x, const char **strs);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::methoh' has {{.*}} 'Mother::method'

  virtual Child *creat(int i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::creat' has {{.*}} 'Father::create'

  virtual Derived &&generat();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::generat' has {{.*}} 'Father::generate'

private:
  void funk(); // Should not warn: access qualifers don't match.
};
