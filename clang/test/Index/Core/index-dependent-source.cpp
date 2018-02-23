// RUN: c-index-test core -print-source-symbols -- %s -std=c++14 -target x86_64-apple-macosx10.7 | FileCheck %s

int invalid;

class Base {
  void baseFunction();

  int baseField;

  static void staticBaseFunction();
};

template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();

  T baseTemplateField;

  static T baseTemplateVariable;
};

template<typename T, typename S>
class TemplateClass: public Base , public BaseTemplate<T> {
public:
  ~TemplateClass();

  T function() { }

  static void staticFunction() { }

  T field;

  static T variable;

  struct Struct { };

  enum Enum { EnumValue };

  using TypeAlias = S;
  typedef T Typedef;

  void overload1(const T &);
  void overload1(const S &);
};

template<typename T, typename S>
void indexSimpleDependentDeclarations(const TemplateClass<T, S> &object) {
  // Valid instance members:
  object.function();
// CHECK: [[@LINE-1]]:10 | instance-method/C++ | function | c:@ST>2#T#T@TemplateClass@F@function# | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1
  object.field;
// CHECK: [[@LINE-1]]:10 | field/C++ | field | c:@ST>2#T#T@TemplateClass@FI@field | <no-cgname> | Ref,RelCont | rel: 1
  object.baseFunction();
// CHECK: [[@LINE-1]]:10 | instance-method/C++ | baseFunction | c:@S@Base@F@baseFunction# | __ZN4Base12baseFunctionEv | Ref,Call,RelCall,RelCont | rel: 1
  object.baseField;
// CHECK: [[@LINE-1]]:10 | field/C++ | baseField | c:@S@Base@FI@baseField | <no-cgname> | Ref,RelCont | rel: 1
  object.baseTemplateFunction();
// CHECK: [[@LINE-1]]:10 | instance-method/C++ | baseTemplateFunction | c:@ST>1#T@BaseTemplate@F@baseTemplateFunction# | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1
  object.baseTemplateField;
// CHECK: [[@LINE-1]]:10 | field/C++ | baseTemplateField | c:@ST>1#T@BaseTemplate@FI@baseTemplateField | <no-cgname> | Ref,RelCont | rel: 1

  // Invalid instance members:
  object.variable;
// CHECK-NOT: [[@LINE-1]]:10
  object.staticFunction();
// CHECK-NOT: [[@LINE-1]]:10
  object.Struct;
// CHECK-NOT: [[@LINE-1]]:10
  object.EnumValue;
// CHECK-NOT: [[@LINE-1]]:10

  // Valid static members:
  TemplateClass<T, S>::staticFunction();
// CHECK: [[@LINE-1]]:24 | static-method/C++ | staticFunction | c:@ST>2#T#T@TemplateClass@F@staticFunction#S | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1
  TemplateClass<T, S>::variable;
// CHECK: [[@LINE-1]]:24 | static-property/C++ | variable | c:@ST>2#T#T@TemplateClass@variable | __ZN13TemplateClass8variableE | Ref,RelCont | rel: 1
  TemplateClass<T, S>::staticBaseFunction();
// CHECK: [[@LINE-1]]:24 | static-method/C++ | staticBaseFunction | c:@S@Base@F@staticBaseFunction#S | __ZN4Base18staticBaseFunctionEv | Ref,Call,RelCall,RelCont | rel: 1
  TemplateClass<T, S>::baseTemplateVariable;
// CHECK: [[@LINE-1]]:24 | static-property/C++ | baseTemplateVariable | c:@ST>1#T@BaseTemplate@baseTemplateVariable | __ZN12BaseTemplate20baseTemplateVariableE | Ref,RelCont | rel: 1
  TemplateClass<T, S>::EnumValue;
// CHECK: [[@LINE-1]]:24 | enumerator/C | EnumValue | c:@ST>2#T#T@TemplateClass@E@Enum@EnumValue | <no-cgname> | Ref,RelCont | rel: 1
  TemplateClass<T, S>::Struct();
// CHECK: [[@LINE-1]]:24 | struct/C | Struct | c:@ST>2#T#T@TemplateClass@S@Struct | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1

  // Invalid static members:
  TemplateClass<T, S>::field;
// CHECK-NOT: [[@LINE-1]]:24
  TemplateClass<T, S>::function();
// CHECK-NOT: [[@LINE-1]]:24

  // Valid type names:
  typename TemplateClass<T, S>::Struct Val;
// CHECK: [[@LINE-1]]:33 | struct/C | Struct | c:@ST>2#T#T@TemplateClass@S@Struct | <no-cgname> | Ref,RelCont | rel: 1
  typename TemplateClass<T, S>::Enum EnumVal;
// CHECK: [[@LINE-1]]:33 | enum/C | Enum | c:@ST>2#T#T@TemplateClass@E@Enum | <no-cgname> | Ref,RelCont | rel: 1
  typename TemplateClass<T, S>::TypeAlias Val2;
// CHECK: [[@LINE-1]]:33 | type-alias/C++ | TypeAlias | c:@ST>2#T#T@TemplateClass@TypeAlias | <no-cgname> | Ref,RelCont | rel: 1
  typename TemplateClass<T, S>::Typedef Val3;
// CHECK: [[@LINE-1]]:33 | type-alias/C | Typedef | c:{{.*}}index-dependent-source.cpp@ST>2#T#T@TemplateClass@T@Typedef | <no-cgname> | Ref,RelCont | rel: 1

  // Invalid type names:
  typename TemplateClass<T, S>::field Val4;
// CHECK-NOT: [[@LINE-1]]:33
  typename TemplateClass<T, S>::staticFunction Val5;
// CHECK-NOT: [[@LINE-1]]:33


  object.invalid;
// CHECK-NOT: [[@LINE-1]]:10
  TemplateClass<T, S>::invalid;
// CHECK-NOT: [[@LINE-1]]:24
}

template<typename T, typename S, typename Y>
void indexDependentOverloads(const TemplateClass<T, S> &object) {
  object.overload1(T());
// CHECK-NOT: [[@LINE-1]]
  object.overload1(S());
// CHECK-NOT: [[@LINE-1]]
  object.overload1(Y());
// CHECK-NOT: [[@LINE-1]]
}

template<typename T> struct UndefinedTemplateClass;

template<typename T>
void undefinedTemplateLookup(UndefinedTemplateClass<T> &x) {
// Shouldn't crash!
  x.lookup;
  typename UndefinedTemplateClass<T>::Type y;
}

template<typename T>
struct UserOfUndefinedTemplateClass: UndefinedTemplateClass<T> { };

template<typename T>
void undefinedTemplateLookup2(UserOfUndefinedTemplateClass<T> &x) {
// Shouldn't crash!
  x.lookup;
  typename UserOfUndefinedTemplateClass<T>::Type y;
}

template<typename T> struct Dropper;

template<typename T> struct Trait;

template<typename T>
struct Recurse : Trait<typename Dropper<T>::Type> { };

template<typename T>
struct Trait : Recurse<T> {
};

template<typename T>
void infiniteTraitRecursion(Trait<T> &t) {
// Shouldn't crash!
  t.lookup;
}
