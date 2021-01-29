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

template <typename T>
struct UsingA {
// CHECK: [[@LINE+1]]:15 | type-alias/C | Type | c:index-dependent-source.cpp@ST>1#T@UsingA@T@Type | <no-cgname> | Def,RelChild | rel: 1
  typedef int Type;
// CHECK: [[@LINE+1]]:15 | static-method/C++ | func | c:@ST>1#T@UsingA@F@func#S | <no-cgname> | Decl,RelChild | rel: 1
  static void func();
// CHECK: [[@LINE+1]]:8 | instance-method/C++ | operator() | c:@ST>1#T@UsingA@F@operator()#I# | <no-cgname> | Decl,RelChild | rel: 1
  void operator()(int);
// CHECK: [[@LINE+1]]:8 | instance-method/C++ | operator+ | c:@ST>1#T@UsingA@F@operator+#&1>@ST>1#T@UsingA1t0.0# | <no-cgname> | Decl,RelChild | rel: 1
  void operator+(const UsingA &);
};

template <typename T>
struct OtherUsing {};

template <typename T>
struct UsingB : public UsingA<T> {
// CHECK: [[@LINE+2]]:40 | type-alias/C | TypeB | c:index-dependent-source.cpp@ST>1#T@UsingB@T@TypeB | <no-cgname> | Def,RelChild | rel: 1
// CHECK: [[@LINE+1]]:20 | struct(Gen)/C++ | OtherUsing | c:@ST>1#T@OtherUsing | <no-cgname> | Ref,RelCont | rel: 1
  typedef typename OtherUsing<T>::Type TypeB;
// CHECK: [[@LINE+2]]:29 | using/using-typename(Gen)/C++ | Type | c:index-dependent-source.cpp@ST>1#T@UsingB@UUT@UsingA<T>::Type | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:18 | struct(Gen)/C++ | UsingA | c:@ST>1#T@UsingA | <no-cgname> | Ref,RelCont | rel: 1
  using typename UsingA<T>::Type;
// CHECK: [[@LINE+2]]:20 | using/using-value(Gen)/C++ | func | c:index-dependent-source.cpp@ST>1#T@UsingB@UUV@UsingA<T>::func | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:9 | struct(Gen)/C++ | UsingA | c:@ST>1#T@UsingA | <no-cgname> | Ref,RelCont | rel: 1
  using UsingA<T>::func;

// CHECK: [[@LINE+2]]:20 | using/using-value(Gen)/C++ | operator() | c:index-dependent-source.cpp@ST>1#T@UsingB@UUV@UsingA<T>::operator() | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:9 | struct(Gen)/C++ | UsingA | c:@ST>1#T@UsingA | <no-cgname> | Ref,RelCont | rel: 1
  using UsingA<T>::operator();
// CHECK: [[@LINE+2]]:20 | using/using-value(Gen)/C++ | operator+ | c:index-dependent-source.cpp@ST>1#T@UsingB@UUV@UsingA<T>::operator+ | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:9 | struct(Gen)/C++ | UsingA | c:@ST>1#T@UsingA | <no-cgname> | Ref,RelCont | rel: 1
  using UsingA<T>::operator+;
};

template <typename T>
struct UsingC : public UsingB<T> {
  static void test() {
// CHECK: [[@LINE+2]]:25 | type-alias/C | TypeB | c:index-dependent-source.cpp@ST>1#T@UsingB@T@TypeB | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE+1]]:14 | struct(Gen)/C++ | UsingB | c:@ST>1#T@UsingB | <no-cgname> | Ref,RelCont | rel: 1
    typename UsingB<T>::TypeB value1;
// CHECK: [[@LINE+2]]:25 | using/using-typename(Gen)/C++ | Type | c:index-dependent-source.cpp@ST>1#T@UsingB@UUT@UsingA<T>::Type | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE+1]]:14 | struct(Gen)/C++ | UsingB | c:@ST>1#T@UsingB | <no-cgname> | Ref,RelCont | rel: 1
    typename UsingB<T>::Type value2;
// CHECK: [[@LINE+2]]:16 | using/using-value(Gen)/C++ | func | c:index-dependent-source.cpp@ST>1#T@UsingB@UUV@UsingA<T>::func | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1
// CHECK: [[@LINE+1]]:5 | struct(Gen)/C++ | UsingB | c:@ST>1#T@UsingB | <no-cgname> | Ref,RelCont | rel: 1
    UsingB<T>::func();
  }
};

template <typename T>
struct UsingD {
// CHECK: [[@LINE+1]]:8 | instance-method/C++ | foo | c:@ST>1#T@UsingD@F@foo#t0.0# | <no-cgname> | Decl,RelChild | rel: 1
  void foo(T);
};

template <typename T, typename U>
struct UsingE : public UsingD<T>, public UsingD<U> {
// CHECK: [[@LINE+2]]:20 | using/using-value(Gen)/C++ | foo | c:index-dependent-source.cpp@ST>2#T#T@UsingE@UUV@UsingD<T>::foo | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:9 | struct(Gen)/C++ | UsingD | c:@ST>1#T@UsingD | <no-cgname> | Ref,RelCont | rel: 1
  using UsingD<T>::foo;
// CHECK: [[@LINE+2]]:20 | using/using-value(Gen)/C++ | foo | c:index-dependent-source.cpp@ST>2#T#T@UsingE@UUV@UsingD<U>::foo | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:9 | struct(Gen)/C++ | UsingD | c:@ST>1#T@UsingD | <no-cgname> | Ref,RelCont | rel: 1
  using UsingD<U>::foo;
};

template <typename T> void foo();
// CHECK: [[@LINE-1]]:28 | function/C | foo | c:@FT@>1#Tfoo#v# | <no-cgname> | Decl | rel: 0
template <typename T> void bar() {
  foo<T>();
// CHECK: [[@LINE-1]]:3 | function/C | foo | c:@FT@>1#Tfoo#v# | <no-cgname> | Ref,Call,RelCall,RelCont | rel: 1
}
