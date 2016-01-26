// RUN: %check_clang_tidy %s misc-virtual-near-miss %t

struct Base {
  virtual void func();
  virtual void gunk();
  virtual ~Base();
  virtual Base &operator=(const Base &);
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

  Derived &operator==(const Base &); // Should not warn: operators are ignored.
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
  virtual int decay(const char *str);
};

class Child : private Father, private Mother {
public:
  Child();

  virtual void func2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::func2' has {{.*}} 'Father::func'

  int methoe(int x, char **strs); // Should not warn: parameter types don't match.

  int methoe(int x);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::methoe' has {{.*}} 'Mother::method'

  void methof(int x, const char **strs); // Should not warn: return types don't match.

  int methoh(int x, const char **strs);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::methoh' has {{.*}} 'Mother::method'

  virtual Child *creat(int i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::creat' has {{.*}} 'Father::create'

  virtual Derived &&generat();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::generat' has {{.*}} 'Father::generate'

  int decaz(const char str[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::decaz' has {{.*}} 'Mother::decay'

  operator bool();
private:
  void funk();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::funk' has {{.*}} 'Father::func'
};
