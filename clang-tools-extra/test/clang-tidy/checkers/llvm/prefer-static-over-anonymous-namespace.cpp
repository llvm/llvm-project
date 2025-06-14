// RUN: %check_clang_tidy %s llvm-prefer-static-over-anonymous-namespace %t -- -- -fno-delayed-template-parsing

namespace {

void regularFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'regularFunction' is declared in an anonymous namespace; prefer using 'static' for restricting visibility [llvm-prefer-static-over-anonymous-namespace]

void declaredFunction();

static void staticFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: place static function 'staticFunction' outside of an anonymous namespace

int globalVariable = 42;

static int staticVariable = 42;

class MyClass {
public:
  MyClass();
  MyClass(const MyClass&) {}
  MyClass(MyClass&&) = default;
  MyClass& operator=(const MyClass&);
  MyClass& operator=(MyClass&&);
  bool operator<(const MyClass&) const;
  void memberFunction();
  static void staticMemberFunction();
  void memberDefinedInClass() {}
  static void staticMemberDefinedInClass() {}
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

template<typename T>
void templateFunction(T Value) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'templateFunction' is declared in an anonymous namespace;

template<>
void templateFunction<int>(int Value) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'templateFunction<int>' is declared in an anonymous namespace;

template<typename T>
class TemplateClass {
public:
  TemplateClass();
  TemplateClass(const TemplateClass&) {}
  TemplateClass(TemplateClass&&) = default;
  TemplateClass& operator=(const TemplateClass&);
  TemplateClass& operator=(TemplateClass&&);
  bool operator<(const TemplateClass&) const;
  void memberFunc();
  T getValue() const;
  void memberDefinedInClass() {}
  static void staticMemberDefinedInClass() {}
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
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function 'inlineFunction' is declared in an anonymous namespace;

auto autoReturnFunction() -> int { return 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'autoReturnFunction' is declared in an anonymous namespace;

class OuterClass {
public:
  class NestedClass {
  public:
    void nestedMemberFunc();
  };
};

void OuterClass::NestedClass::nestedMemberFunc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: place definition of method 'nestedMemberFunc' outside of an anonymous namespace


// Variables are not warned by default
template<typename T>
constexpr T Pi = T(3.1415926);

void (*funcPtr)() = nullptr;

auto lambda = []() { return 42; };

} // namespace

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
