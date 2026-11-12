// RUN: %check_clang_tidy %s llvm-prefer-static-over-anonymous-namespace %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=,VAR %s llvm-prefer-static-over-anonymous-namespace %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     llvm-prefer-static-over-anonymous-namespace.AllowVariableDeclarations: false }, \
// RUN:   }" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=,MEM %s llvm-prefer-static-over-anonymous-namespace %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     llvm-prefer-static-over-anonymous-namespace.AllowMemberFunctionsInClass: false }, \
// RUN:   }" -- -fno-delayed-template-parsing

namespace {

void regularFunction() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'regularFunction' is declared in an anonymous namespace; prefer using 'static' for restricting visibility [llvm-prefer-static-over-anonymous-namespace]

  int Variable = 42;
  auto Lambda = []() { return 42; };
  static int StaticVariable = 42;
}

void declaredFunction();

static void staticFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: place static function 'staticFunction' outside of an anonymous namespace

int globalVariable = 42;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:5: warning: variable 'globalVariable' is declared in an anonymous namespace;

static int staticVariable = 42;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:12: warning: place static variable 'staticVariable' outside of an anonymous namespace

typedef int MyInt;
const MyInt myGlobalVariable = 42;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:13: warning: variable 'myGlobalVariable' is declared in an anonymous namespace;

template<typename T>
constexpr T Pi = T(3.1415926);
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:13: warning: variable 'Pi' is declared in an anonymous namespace;

void (*funcPtr)() = nullptr;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:8: warning: variable 'funcPtr' is declared in an anonymous namespace;

auto lambda = []() { return 42; };
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:6: warning: variable 'lambda' is declared in an anonymous namespace;

class InstanceClass {
  int member;
};

InstanceClass instance;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:15: warning: variable 'instance' is declared in an anonymous namespace;

InstanceClass* instancePtr = nullptr;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:16: warning: variable 'instancePtr' is declared in an anonymous namespace;

InstanceClass& instanceRef = instance;
// CHECK-MESSAGES-VAR: :[[@LINE-1]]:16: warning: variable 'instanceRef' is declared in an anonymous namespace;

class MyClass {
public:
  MyClass();
  MyClass(const MyClass&) {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'MyClass' outside of an anonymous namespace
  MyClass(MyClass&&) = default;
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'MyClass' outside of an anonymous namespace
  MyClass& operator=(const MyClass&);
  MyClass& operator=(MyClass&&);
  bool operator<(const MyClass&) const;
  void memberFunction();
  static void staticMemberFunction();
  void memberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
  static void staticMemberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:15: warning: place definition of method 'staticMemberDefinedInClass' outside of an anonymous namespace
  template <typename T>
  void templateFunction();
  template <typename T>
  void templateFunctionInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'templateFunctionInClass' outside of an anonymous namespace
};

MyClass::MyClass() {}
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: place definition of method 'MyClass' outside of an anonymous namespace

MyClass& MyClass::operator=(const MyClass&) { return *this; }
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: place definition of method 'operator=' outside of an anonymous namespace

MyClass& MyClass::operator=(MyClass&&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: place definition of method 'operator=' outside of an anonymous namespace

bool MyClass::operator<(const MyClass&) const { return true; }
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: place definition of method 'operator<' outside of an anonymous namespace

void MyClass::memberFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: place definition of method 'memberFunction' outside of an anonymous namespace

void MyClass::staticMemberFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: place definition of method 'staticMemberFunction' outside of an anonymous namespace

template <typename T>
void MyClass::templateFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: place definition of method 'templateFunction' outside of an anonymous namespace

template<typename T>
void templateFunction(T Value) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'templateFunction' is declared in an anonymous namespace; prefer using 'static' for restricting visibility

template<>
void templateFunction<int>(int Value) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'templateFunction<int>' is declared in an anonymous namespace; prefer using 'static' for restricting visibility

template<typename T>
class TemplateClass {
public:
  TemplateClass();
  TemplateClass(const TemplateClass&) {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClass<T>' outside of an anonymous namespace
  TemplateClass(TemplateClass&&) = default;
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClass<T>' outside of an anonymous namespace
  TemplateClass& operator=(const TemplateClass&);
  TemplateClass& operator=(TemplateClass&&);
  bool operator<(const TemplateClass&) const;
  void memberFunc();
  T getValue() const;
  void memberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
  static void staticMemberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:15: warning: place definition of method 'staticMemberDefinedInClass' outside of an anonymous namespace
  template <typename U>
  void templateMethodInTemplateClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'templateMethodInTemplateClass' outside of an anonymous namespace
private:
  T Value;
};

template<typename T>
TemplateClass<T>::TemplateClass() {}
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: place definition of method 'TemplateClass<T>' outside of an anonymous namespace

template<typename T>
TemplateClass<T>& TemplateClass<T>::operator=(const TemplateClass&) { return *this; }
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: place definition of method 'operator=' outside of an anonymous namespace

template<typename T>
TemplateClass<T>& TemplateClass<T>::operator=(TemplateClass&&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: place definition of method 'operator=' outside of an anonymous namespace

template<typename T>
bool TemplateClass<T>::operator<(const TemplateClass&) const { return true; }
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: place definition of method 'operator<' outside of an anonymous namespace

template<typename T>
void TemplateClass<T>::memberFunc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: place definition of method 'memberFunc' outside of an anonymous namespace

template<typename T>
T TemplateClass<T>::getValue() const { return Value; }
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: place definition of method 'getValue' outside of an anonymous namespace

inline void inlineFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function 'inlineFunction' is declared in an anonymous namespace; prefer using 'static' for restricting visibility

auto autoReturnFunction() -> int { return 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'autoReturnFunction' is declared in an anonymous namespace; prefer using 'static' for restricting visibility

class OuterClass {
public:
  class NestedClass {
  public:
    void nestedMemberFunc();
    void nestedMemberDefinedInClass() {}
    // CHECK-MESSAGES-MEM: :[[@LINE-1]]:10: warning: place definition of method 'nestedMemberDefinedInClass' outside of an anonymous namespace
  };
};

void OuterClass::NestedClass::nestedMemberFunc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: place definition of method 'nestedMemberFunc' outside of an anonymous namespace

} // namespace

namespace {

class MyClassOutOfAnon {
public:
  MyClassOutOfAnon();
  MyClassOutOfAnon(const MyClassOutOfAnon&) {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'MyClassOutOfAnon' outside of an anonymous namespace
  MyClassOutOfAnon(MyClassOutOfAnon&&) = default;
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'MyClassOutOfAnon' outside of an anonymous namespace
  MyClassOutOfAnon& operator=(const MyClassOutOfAnon&);
  MyClassOutOfAnon& operator=(MyClassOutOfAnon&&);
  bool operator<(const MyClassOutOfAnon&) const;
  void memberFunction();
  static void staticMemberFunction();
  void memberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
  static void staticMemberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:15: warning: place definition of method 'staticMemberDefinedInClass' outside of an anonymous namespace
  template <typename T>
  void templateFunction();
  template <typename T>
  void templateFunctionInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'templateFunctionInClass' outside of an anonymous namespace
};

} // namespace

MyClassOutOfAnon::MyClassOutOfAnon() {}

MyClassOutOfAnon& MyClassOutOfAnon::operator=(const MyClassOutOfAnon&) { return *this; }

MyClassOutOfAnon& MyClassOutOfAnon::operator=(MyClassOutOfAnon&&) = default;

bool MyClassOutOfAnon::operator<(const MyClassOutOfAnon&) const { return true; }

void MyClassOutOfAnon::memberFunction() {}

void MyClassOutOfAnon::staticMemberFunction() {}

template <typename T>
void MyClassOutOfAnon::templateFunction() {}

namespace {

template<typename T>
class TemplateClassOutOfAnon {
  public:
  TemplateClassOutOfAnon();
  TemplateClassOutOfAnon(const TemplateClassOutOfAnon&) {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClassOutOfAnon<T>' outside of an anonymous namespace
  TemplateClassOutOfAnon(TemplateClassOutOfAnon&&) = default;
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClassOutOfAnon<T>' outside of an anonymous namespace
  TemplateClassOutOfAnon& operator=(const TemplateClassOutOfAnon&);
  TemplateClassOutOfAnon& operator=(TemplateClassOutOfAnon&&);
  bool operator<(const TemplateClassOutOfAnon&) const;
  void memberFunc();
  T getValue() const;
  void memberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
  static void staticMemberDefinedInClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:15: warning: place definition of method 'staticMemberDefinedInClass' outside of an anonymous namespace
  template <typename U>
  void templateMethodInTemplateClass() {}
  // CHECK-MESSAGES-MEM: :[[@LINE-1]]:8: warning: place definition of method 'templateMethodInTemplateClass' outside of an anonymous namespace
  private:
  T Value;
};

} // namespace

template<typename T>
TemplateClassOutOfAnon<T>::TemplateClassOutOfAnon() {}

template<typename T>
TemplateClassOutOfAnon<T>& TemplateClassOutOfAnon<T>::operator=(const TemplateClassOutOfAnon&) { return *this; }

template<typename T>
TemplateClassOutOfAnon<T>& TemplateClassOutOfAnon<T>::operator=(TemplateClassOutOfAnon&&) = default;

template<typename T>
bool TemplateClassOutOfAnon<T>::operator<(const TemplateClassOutOfAnon&) const { return true; }

template<typename T>
void TemplateClassOutOfAnon<T>::memberFunc() {}

template<typename T>
T TemplateClassOutOfAnon<T>::getValue() const { return Value; }


#define DEFINE_FUNCTION(name) \
  namespace { \
    void name() {} \
  }

DEFINE_FUNCTION(macroDefinedFunction)

#define DECLARE_VAR(type, name, value) \
  namespace { \
    type name = value; \
  }

DECLARE_VAR(int, macroVariable, 42)

namespace {

#define INTERNAL_FUNC void internalMacroFunc() {}

INTERNAL_FUNC

} // namespace
