// RUN: %check_clang_tidy %s llvm-prefer-static-over-anonymous-namespace %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     llvm-prefer-static-over-anonymous-namespace.AllowMemberFunctionsInClass: false }, \
// RUN:   }" -- -fno-delayed-template-parsing

namespace {

class MyClass {
public:
  MyClass() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: place definition of method 'MyClass' outside of an anonymous namespace
  MyClass(const MyClass&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: place definition of method 'MyClass' outside of an anonymous namespace
  MyClass& operator=(const MyClass&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: place definition of method 'operator=' outside of an anonymous namespace
  MyClass& operator=(MyClass&&) { return *this; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: place definition of method 'operator=' outside of an anonymous namespace
  bool operator<(const MyClass&) const { return true; };
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: place definition of method 'operator<' outside of an anonymous namespace
  void memberFunction();
  void memberDefinedInClass() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
};

void MyClass::memberFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: place definition of method 'memberFunction' outside of an anonymous namespace

template<typename T>
class TemplateClass {
public:
  TemplateClass() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClass<T>' outside of an anonymous namespace
  TemplateClass(const TemplateClass&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: place definition of method 'TemplateClass<T>' outside of an anonymous namespace
  TemplateClass& operator=(const TemplateClass&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: place definition of method 'operator=' outside of an anonymous namespace
  TemplateClass& operator=(TemplateClass&&) { return *this; };
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: place definition of method 'operator=' outside of an anonymous namespace
  bool operator<(const TemplateClass&) const { return true; };
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: place definition of method 'operator<' outside of an anonymous namespace
  void memberFunc();
  T getValue() const { return {}; };
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: place definition of method 'getValue' outside of an anonymous namespace
  void memberDefinedInClass() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: place definition of method 'memberDefinedInClass' outside of an anonymous namespace
};

template<typename T>
void TemplateClass<T>::memberFunc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: place definition of method 'memberFunc' outside of an anonymous namespace

class OuterClass {
public:
  class NestedClass {
  public:
    void nestedMemberFunc();
    void nestedMemberDefinedInClass() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: place definition of method 'nestedMemberDefinedInClass' outside of an anonymous namespace
  };
};

void OuterClass::NestedClass::nestedMemberFunc() {}
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: place definition of method 'nestedMemberFunc' outside of an anonymous namespace

} // namespace
