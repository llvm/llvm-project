int invalid;

class Base {
  void baseFunction(); // CHECK3: [[@LINE]]:8 -> [[@LINE]]:20

  int baseField;

  static void staticBaseFunction(); // CHECK7: [[@LINE]]:15 -> [[@LINE]]:33
};

template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();

  T baseTemplateField; // CHECK4: [[@LINE]]:5 -> [[@LINE]]:22

  static T baseTemplateVariable; // CHECK8: [[@LINE]]:12 -> [[@LINE]]:32
};

template<typename T, typename S>
class TemplateClass: public Base , public BaseTemplate<T> {
public:
  ~TemplateClass();

  T function() { } // CHECK1: [[@LINE]]:5 -> [[@LINE]]:13

  static void staticFunction() { } // CHECK5: [[@LINE]]:15 -> [[@LINE]]:29

  T field; // CHECK2: [[@LINE]]:5 -> [[@LINE]]:10

  static T variable; // CHECK6: [[@LINE]]:12 -> [[@LINE]]:20

  struct Struct { }; // CHECK9: [[@LINE]]:10 -> [[@LINE]]:16

  enum Enum { EnumValue }; // CHECK10: [[@LINE]]:8 -> [[@LINE]]:12

  using TypeAlias = S; // CHECK11: [[@LINE]]:9 -> [[@LINE]]:18
  typedef T Typedef;

  void overload1(const T &);
  void overload1(const S &);
};

template<typename T, typename S>
void renameSimpleDependentDeclarations(const TemplateClass<T, S> &object) {

  object.function(); // CHECK1: [[@LINE]]:10 -> [[@LINE]]:18
// RUN: clang-refactor-test rename-initiate -at=%s:26:5 -at=%s:48:10 -new-name=x %s | FileCheck --check-prefix=CHECK1 %s
  object.field; // CHECK2: [[@LINE]]:10 -> [[@LINE]]:15
// RUN: clang-refactor-test rename-initiate -at=%s:30:5 -at=%s:50:10 -new-name=x %s | FileCheck --check-prefix=CHECK2 %s
  object.baseFunction(); // CHECK3: [[@LINE]]:10 -> [[@LINE]]:22
// RUN: clang-refactor-test rename-initiate -at=%s:4:8 -at=%s:52:10 -new-name=x %s | FileCheck --check-prefix=CHECK3 %s
  object.baseTemplateField; // CHECK4: [[@LINE]]:10 -> [[@LINE]]:27
// RUN: clang-refactor-test rename-initiate -at=%s:16:5 -at=%s:54:10 -new-name=x %s | FileCheck --check-prefix=CHECK4 %s

  TemplateClass<T, S>::staticFunction(); // CHECK5: [[@LINE]]:24 -> [[@LINE]]:38
// RUN: clang-refactor-test rename-initiate -at=%s:28:15 -at=%s:57:24 -new-name=x %s | FileCheck --check-prefix=CHECK5 %s
  TemplateClass<T, S>::variable; // CHECK6: [[@LINE]]:24 -> [[@LINE]]:32
// RUN: clang-refactor-test rename-initiate -at=%s:32:12 -at=%s:59:24 -new-name=x %s | FileCheck --check-prefix=CHECK6 %s
  TemplateClass<T, S>::staticBaseFunction(); // CHECK7: [[@LINE]]:24 -> [[@LINE]]:42
// RUN: clang-refactor-test rename-initiate -at=%s:8:15 -at=%s:61:24 -new-name=x %s | FileCheck --check-prefix=CHECK7 %s
  TemplateClass<T, S>::baseTemplateVariable; // CHECK8: [[@LINE]]:24 -> [[@LINE]]:44
// RUN: clang-refactor-test rename-initiate -at=%s:18:12 -at=%s:63:24 -new-name=x %s | FileCheck --check-prefix=CHECK8 %s

  typename TemplateClass<T, S>::Struct Val; // CHECK9: [[@LINE]]:33 -> [[@LINE]]:39
// RUN: clang-refactor-test rename-initiate -at=%s:34:10 -at=%s:66:33 -new-name=x %s | FileCheck --check-prefix=CHECK9 %s
  typename TemplateClass<T, S>::Enum EnumVal; // CHECK10: [[@LINE]]:33 -> [[@LINE]]:37
// RUN: clang-refactor-test rename-initiate -at=%s:36:8 -at=%s:68:33 -new-name=x %s | FileCheck --check-prefix=CHECK10 %s
  typename TemplateClass<T, S>::TypeAlias Val2; // CHECK11: [[@LINE]]:33 -> [[@LINE]]:42
// RUN: clang-refactor-test rename-initiate -at=%s:38:9 -at=%s:70:33 -new-name=x %s | FileCheck --check-prefix=CHECK11 %s
}
