// RUN: %clang_cc1 -fno-elide-constructors -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s
// RUN: %clang_cc1 -fno-elide-constructors -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | opt -verify
// FIXME: remove the call to "opt" once the tests are running the Clang verifier automatically again.

int Bar(int);
int Baz(int);

int Func1(int x) {
  if (x) {
    // CHECK: %call = musttail call noundef i32 @_Z3Bari(i32 noundef %1)
    // CHECK-NEXT: ret i32 %call
    [[clang::musttail]] return Bar(x);
  } else {
    [[clang::musttail]] return Baz(x); // CHECK: %call1 = musttail call noundef i32 @_Z3Bazi(i32 noundef %3)
  }
}

int Func2(int x) {
  {
    [[clang::musttail]] return Bar(Bar(x));
  }
}

// CHECK: %call1 = musttail call noundef i32 @_Z3Bari(i32 noundef %call)

class Foo {
public:
  static int StaticMethod(int x);
  int MemberFunction(int x);
  int TailFrom(int x);
  int TailFrom2(int x);
  int TailFrom3(int x);
};

int Foo::TailFrom(int x) {
  [[clang::musttail]] return MemberFunction(x);
}

// CHECK: %call = musttail call noundef i32 @_ZN3Foo14MemberFunctionEi(%class.Foo* noundef %this1, i32 noundef %0)

int Func3(int x) {
  [[clang::musttail]] return Foo::StaticMethod(x);
}

// CHECK: %call = musttail call noundef i32 @_ZN3Foo12StaticMethodEi(i32 noundef %0)

int Func4(int x) {
  Foo foo; // Object with trivial destructor.
  [[clang::musttail]] return foo.StaticMethod(x);
}

// CHECK: %call = musttail call noundef i32 @_ZN3Foo12StaticMethodEi(i32 noundef %0)

int (Foo::*pmf)(int);

int Foo::TailFrom2(int x) {
  [[clang::musttail]] return ((*this).*pmf)(x);
}

// CHECK: %call = musttail call noundef i32 %8(%class.Foo* noundef %this.adjusted, i32 noundef %9)

int Foo::TailFrom3(int x) {
  [[clang::musttail]] return (this->*pmf)(x);
}

// CHECK: %call = musttail call noundef i32 %8(%class.Foo* noundef %this.adjusted, i32 noundef %9)

void ReturnsVoid();

void Func5() {
  [[clang::musttail]] return ReturnsVoid();
}

// CHECK: musttail call void @_Z11ReturnsVoidv()

class HasTrivialDestructor {};

int ReturnsInt(int x);

int Func6(int x) {
  HasTrivialDestructor foo;
  [[clang::musttail]] return ReturnsInt(x);
}

// CHECK: %call = musttail call noundef i32 @_Z10ReturnsInti(i32 noundef %0)

struct Data {
  int (*fptr)(Data *);
};

int Func7(Data *data) {
  [[clang::musttail]] return data->fptr(data);
}

// CHECK: %call = musttail call noundef i32 %1(%struct.Data* noundef %2)

template <class T>
T TemplateFunc(T) {
  return 5;
}

int Func9(int x) {
  [[clang::musttail]] return TemplateFunc<int>(x);
}

// CHECK: %call = musttail call noundef i32 @_Z12TemplateFuncIiET_S0_(i32 noundef %0)

template <class T>
int Func10(int x) {
  T t;
  [[clang::musttail]] return Bar(x);
}

int Func11(int x) {
  return Func10<int>(x);
}

// CHECK: %call = musttail call noundef i32 @_Z3Bari(i32 noundef %0)

template <class T>
T Func12(T x) {
  [[clang::musttail]] return ::Bar(x);
}

int Func13(int x) {
  return Func12<int>(x);
}

// CHECK: %call = musttail call noundef i32 @_Z3Bari(i32 noundef %0)

int Func14(int x) {
  int vla[x];
  [[clang::musttail]] return Bar(x);
}

// CHECK: %call = musttail call noundef i32 @_Z3Bari(i32 noundef %3)

void TrivialDestructorParam(HasTrivialDestructor obj);

void Func14(HasTrivialDestructor obj) {
  [[clang::musttail]] return TrivialDestructorParam(obj);
}

// CHECK: musttail call void @_Z22TrivialDestructorParam20HasTrivialDestructor()

struct Struct3 {
  void ConstMemberFunction(const int *) const;
  void NonConstMemberFunction(int *i);
};
void Struct3::NonConstMemberFunction(int *i) {
  // The parameters are not identical, but they are compatible.
  [[clang::musttail]] return ConstMemberFunction(i);
}

// CHECK: musttail call void @_ZNK7Struct319ConstMemberFunctionEPKi(%struct.Struct3* noundef %this1, i32* noundef %0)

struct HasNonTrivialCopyConstructor {
  HasNonTrivialCopyConstructor(const HasNonTrivialCopyConstructor &);
};
HasNonTrivialCopyConstructor ReturnsClassByValue();
HasNonTrivialCopyConstructor TestNonElidableCopyConstructor() {
  [[clang::musttail]] return (((ReturnsClassByValue())));
}

// CHECK: musttail call void @_Z19ReturnsClassByValuev(%struct.HasNonTrivialCopyConstructor* sret(%struct.HasNonTrivialCopyConstructor) align 1 %agg.result)

struct HasNonTrivialCopyConstructor2 {
  // Copy constructor works even if it has extra default params.
  HasNonTrivialCopyConstructor2(const HasNonTrivialCopyConstructor &, int DefaultParam = 5);
};
HasNonTrivialCopyConstructor2 ReturnsClassByValue2();
HasNonTrivialCopyConstructor2 TestNonElidableCopyConstructor2() {
  [[clang::musttail]] return (((ReturnsClassByValue2())));
}

// CHECK: musttail call void @_Z20ReturnsClassByValue2v()

void TestFunctionPointer(int x) {
  void (*p)(int) = nullptr;
  [[clang::musttail]] return p(x);
}

// CHECK: musttail call void %0(i32 noundef %1)

struct LargeWithCopyConstructor {
  LargeWithCopyConstructor(const LargeWithCopyConstructor &);
  char data[32];
};
LargeWithCopyConstructor ReturnsLarge();
LargeWithCopyConstructor TestLargeWithCopyConstructor() {
  [[clang::musttail]] return ReturnsLarge();
}

// CHECK: define dso_local void @_Z28TestLargeWithCopyConstructorv(%struct.LargeWithCopyConstructor* noalias sret(%struct.LargeWithCopyConstructor) align 1 %agg.result)
// CHECK: musttail call void @_Z12ReturnsLargev(%struct.LargeWithCopyConstructor* sret(%struct.LargeWithCopyConstructor) align 1 %agg.result)

using IntFunctionType = int();
IntFunctionType *ReturnsIntFunction();
int TestRValueFunctionPointer() {
  [[clang::musttail]] return ReturnsIntFunction()();
}

// CHECK: musttail call noundef i32 %call()

void(FuncWithParens)() {
  [[clang::musttail]] return FuncWithParens();
}

// CHECK: musttail call void @_Z14FuncWithParensv()

int TestNonCapturingLambda() {
  auto lambda = []() { return 12; };
  [[clang::musttail]] return (+lambda)();
}

// CHECK: %call = call noundef i32 ()* @"_ZZ22TestNonCapturingLambdavENK3$_0cvPFivEEv"(%class.anon* noundef %lambda)
// CHECK: musttail call noundef i32 %call()

class TestVirtual {
  virtual void TailTo();
  virtual void TailFrom();
};

void TestVirtual::TailFrom() {
  [[clang::musttail]] return TailTo();
}

// CHECK: musttail call void %1(%class.TestVirtual* noundef %this1)
