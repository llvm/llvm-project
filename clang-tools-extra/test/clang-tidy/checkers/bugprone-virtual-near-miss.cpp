// RUN: %check_clang_tidy %s bugprone-virtual-near-miss %t

class NoDefinedClass1;
class NoDefinedClass2;

struct Base {
  virtual void func();
  virtual void gunk();
  virtual ~Base();
  virtual Base &operator=(const Base &);
  virtual NoDefinedClass1 *f();
};

struct Derived : Base {
  // Should not warn "do you want to override 'gunk'?", because gunk is already
  // overriden by this class.
  virtual void funk();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::funk' has a similar name and the same signature as virtual method 'Base::func'; did you mean to override it? [bugprone-virtual-near-miss]
  // CHECK-FIXES: virtual void func();

  void func2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::func2' has {{.*}} 'Base::func'
  // CHECK-FIXES: void func();

  void func22(); // Should not warn.

  void gunk(); // Should not warn: gunk is override.

  void fun();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Derived::fun' has {{.*}} 'Base::func'
  // CHECK-FIXES: void func();

  Derived &operator==(const Base &); // Should not warn: operators are ignored.

  virtual NoDefinedClass2 *f1(); // Should not crash: non-defined class return type is ignored.
};

template <typename T>
struct TBase {
  virtual void tfunc(T t);
};

template <typename T>
struct TDerived : TBase<T> {
  virtual void tfunk(T t);
  // Should not apply fix for template.
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: method 'TDerived::tfunk' has {{.*}} 'TBase::tfunc'
  // CHECK-FIXES: virtual void tfunc(T t);
};

TDerived<int> T1;
TDerived<double> T2;

// Should not fix macro definition
#define MACRO1 void funcM()
// CHECK-FIXES: #define MACRO1 void funcM()
#define MACRO2(m) void m()
// CHECK-FIXES: #define MACRO2(m) void m()

struct DerivedMacro : Base {
  MACRO1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'DerivedMacro::funcM' has {{.*}} 'Base::func'
  // CHECK-FIXES: MACRO1;

  MACRO2(func3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'DerivedMacro::func3' has {{.*}} 'Base::func'
  // CHECK-FIXES: MACRO2(func);
};

typedef Derived derived_type;

class Father {
public:
  Father();
  virtual void func();
  virtual Father *create(int i);
  virtual Base &&generate();
  virtual Base *canonical(Derived D);
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
  // CHECK-FIXES: virtual void func();

  int methoe(int x, char **strs); // Should not warn: parameter types don't match.

  int methoe(int x);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::methoe' has {{.*}} 'Mother::method'
  // CHECK-FIXES: int method(int x);

  void methof(int x, const char **strs); // Should not warn: return types don't match.

  int methoh(int x, const char **strs);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::methoh' has {{.*}} 'Mother::method'
  // CHECK-FIXES: int method(int x, const char **strs);

  virtual Child *creat(int i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::creat' has {{.*}} 'Father::create'
  // CHECK-FIXES: virtual Child *create(int i);

  virtual Derived &&generat();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::generat' has {{.*}} 'Father::generate'
  // CHECK-FIXES: virtual Derived &&generate();

  int decaz(const char str[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::decaz' has {{.*}} 'Mother::decay'
  // CHECK-FIXES: int decay(const char str[]);

  operator bool();

  derived_type *canonica(derived_type D);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::canonica' has {{.*}} 'Father::canonical'
  // CHECK-FIXES: derived_type *canonical(derived_type D);

private:
  void funk();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: method 'Child::funk' has {{.*}} 'Father::func'
  // CHECK-FIXES: void func();
};
