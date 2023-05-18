// RUN: %clang_cc1 -verify -fsyntax-only -fms-extensions -fcxx-exceptions -fopenmp -triple x86_64-linux %s

int ReturnsInt1();
int Func1() {
  [[clang::musttail]] ReturnsInt1();              // expected-error {{'musttail' attribute only applies to return statements}}
  [[clang::musttail(1, 2)]] return ReturnsInt1(); // expected-error {{'musttail' attribute takes no arguments}}
  [[clang::musttail]] return 5;                   // expected-error {{'musttail' attribute requires that the return value is the result of a function call}}
  [[clang::musttail]] return ReturnsInt1();
}

void NoFunctionCall() {
  [[clang::musttail]] return; // expected-error {{'musttail' attribute requires that the return value is the result of a function call}}
}

[[clang::musttail]] static int int_val = ReturnsInt1(); // expected-error {{'musttail' attribute cannot be applied to a declaration}}

void NoParams(); // expected-note {{target function has different number of parameters (expected 1 but has 0)}}
void TestParamArityMismatch(int x) {
  [[clang::musttail]] // expected-note {{tail call required by 'musttail' attribute here}}
  return NoParams();  // expected-error {{cannot perform a tail call to function 'NoParams' because its signature is incompatible with the calling function}}
}

void LongParam(long x); // expected-note {{target function has type mismatch at 1st parameter (expected 'long' but has 'int')}}
void TestParamTypeMismatch(int x) {
  [[clang::musttail]]  // expected-note {{tail call required by 'musttail' attribute here}}
  return LongParam(x); // expected-error {{cannot perform a tail call to function 'LongParam' because its signature is incompatible with the calling function}}
}

long ReturnsLong(); // expected-note {{target function has different return type ('int' expected but has 'long')}}
int TestReturnTypeMismatch() {
  [[clang::musttail]]   // expected-note {{tail call required by 'musttail' attribute here}}
  return ReturnsLong(); // expected-error {{cannot perform a tail call to function 'ReturnsLong' because its signature is incompatible with the calling function}}
}

struct Struct1 {
  void MemberFunction(); // expected-note {{'MemberFunction' declared here}}
};
void TestNonMemberToMember() {
  Struct1 st;
  [[clang::musttail]]         // expected-note {{tail call required by 'musttail' attribute here}}
  return st.MemberFunction(); // expected-error {{non-member function cannot perform a tail call to non-static member function 'MemberFunction'}}
}

void ReturnsVoid(); // expected-note {{'ReturnsVoid' declared here}}
struct Struct2 {
  void TestMemberToNonMember() {
    [[clang::musttail]]   // expected-note {{tail call required by 'musttail' attribute here}}
    return ReturnsVoid(); // expected-error{{non-static member function cannot perform a tail call to non-member function 'ReturnsVoid'}}
  }
};

class HasNonTrivialDestructor {
public:
  ~HasNonTrivialDestructor() {}
  int ReturnsInt();
};

void ReturnsVoid2();
void TestNonTrivialDestructorInScope() {
  HasNonTrivialDestructor foo;              // expected-note {{jump exits scope of variable with non-trivial destructor}}
  [[clang::musttail]] return ReturnsVoid(); // expected-error {{cannot perform a tail call from this return statement}}
}

int NonTrivialParam(HasNonTrivialDestructor x);
int TestNonTrivialParam(HasNonTrivialDestructor x) {
  [[clang::musttail]] return NonTrivialParam(x); // expected-error {{tail call requires that the return value, all parameters, and any temporaries created by the expression are trivially destructible}}
}

HasNonTrivialDestructor ReturnsNonTrivialValue();
HasNonTrivialDestructor TestReturnsNonTrivialValue() {
  // FIXME: the diagnostic cannot currently distinguish between needing to run a
  // destructor for the return value and needing to run a destructor for some
  // other temporary created in the return statement.
  [[clang::musttail]] return (ReturnsNonTrivialValue()); // expected-error {{tail call requires that the return value, all parameters, and any temporaries created by the expression are trivially destructible}}
}

HasNonTrivialDestructor TestReturnsNonTrivialNonFunctionCall() {
  [[clang::musttail]] return HasNonTrivialDestructor(); // expected-error {{'musttail' attribute requires that the return value is the result of a function call}}
}

struct UsesPointerToMember {
  void (UsesPointerToMember::*p_mem)(); // expected-note {{'p_mem' declared here}}
};
void TestUsesPointerToMember(UsesPointerToMember *foo) {
  // "this" pointer cannot double as first parameter.
  [[clang::musttail]]            // expected-note {{tail call required by 'musttail' attribute here}}
  return (foo->*(foo->p_mem))(); // expected-error {{non-member function cannot perform a tail call to pointer-to-member function 'p_mem'}}
}

void ReturnsVoid2();
void TestNestedClass() {
  HasNonTrivialDestructor foo;
  class Nested {
    __attribute__((noinline)) static void NestedMethod() {
      // Outer non-trivial destructor does not affect nested class.
      [[clang::musttail]] return ReturnsVoid2();
    }
  };
}

template <class T>
T TemplateFunc(T x) { // expected-note{{target function has different return type ('long' expected but has 'int')}}
  return x ? 5 : 10;
}
int OkTemplateFunc(int x) {
  [[clang::musttail]] return TemplateFunc<int>(x);
}
template <class T>
T BadTemplateFunc(T x) {
  [[clang::musttail]]          // expected-note {{tail call required by 'musttail' attribute here}}
  return TemplateFunc<int>(x); // expected-error {{cannot perform a tail call to function 'TemplateFunc' because its signature is incompatible with the calling function}}
}
long TestBadTemplateFunc(long x) {
  return BadTemplateFunc<long>(x); // expected-note {{in instantiation of}}
}

void IntParam(int x);
void TestVLA(int x) {
  HasNonTrivialDestructor vla[x];         // expected-note {{jump exits scope of variable with non-trivial destructor}}
  [[clang::musttail]] return IntParam(x); // expected-error {{cannot perform a tail call from this return statement}}
}

void TestNonTrivialDestructorSubArg(int x) {
  [[clang::musttail]] return IntParam(NonTrivialParam(HasNonTrivialDestructor())); // expected-error {{tail call requires that the return value, all parameters, and any temporaries created by the expression are trivially destructible}}
}

void VariadicFunction(int x, ...);
void TestVariadicFunction(int x, ...) {
  [[clang::musttail]] return VariadicFunction(x); // expected-error {{'musttail' attribute may not be used with variadic functions}}
}

int TakesIntParam(int x);     // expected-note {{target function has type mismatch at 1st parameter (expected 'int' but has 'short')}}
int TakesShortParam(short x); // expected-note {{target function has type mismatch at 1st parameter (expected 'short' but has 'int')}}
int TestIntParamMismatch(int x) {
  [[clang::musttail]]        // expected-note {{tail call required by 'musttail' attribute here}}
  return TakesShortParam(x); // expected-error {{cannot perform a tail call to function 'TakesShortParam' because its signature is incompatible with the calling function}}
}
int TestIntParamMismatch2(short x) {
  [[clang::musttail]]      // expected-note {{tail call required by 'musttail' attribute here}}
  return TakesIntParam(x); // expected-error {{cannot perform a tail call to function 'TakesIntParam' because its signature is incompatible with the calling function}}
}

struct TestClassMismatch1 {
  void ToFunction(); // expected-note{{target function is a member of different class (expected 'TestClassMismatch2' but has 'TestClassMismatch1')}}
};
TestClassMismatch1 *tcm1;
struct TestClassMismatch2 {
  void FromFunction() {
    [[clang::musttail]]        // expected-note {{tail call required by 'musttail' attribute here}}
    return tcm1->ToFunction(); // expected-error {{cannot perform a tail call to function 'ToFunction' because its signature is incompatible with the calling function}}
  }
};

__regcall int RegCallReturnsInt(); // expected-note {{target function has calling convention regcall (expected cdecl)}}
int TestMismatchCallingConvention() {
  [[clang::musttail]]         // expected-note {{tail call required by 'musttail' attribute here}}
  return RegCallReturnsInt(); // expected-error {{cannot perform a tail call to function 'RegCallReturnsInt' because it uses an incompatible calling convention}}
}

int TestNonCapturingLambda() {
  auto lambda = []() { return 12; }; // expected-note {{'operator()' declared here}}
  [[clang::musttail]]                // expected-note {{tail call required by 'musttail' attribute here}}
  return lambda();                   // expected-error {{non-member function cannot perform a tail call to non-static member function 'operator()'}}

  // This works.
  auto lambda_fptr = static_cast<int (*)()>(lambda);
  [[clang::musttail]] return lambda_fptr();
  [[clang::musttail]] return (+lambda)();
}

int TestCapturingLambda() {
  int x;
  auto lambda = [x]() { return 12; }; // expected-note {{'operator()' declared here}}
  [[clang::musttail]]                 // expected-note {{tail call required by 'musttail' attribute here}}
  return lambda();                    // expected-error {{non-member function cannot perform a tail call to non-static member function 'operator()'}}
}

int TestNonTrivialTemporary(int) {
  [[clang::musttail]] return TakesIntParam(HasNonTrivialDestructor().ReturnsInt()); // expected-error {{tail call requires that the return value, all parameters, and any temporaries created by the expression are trivially destructible}}
}

void ReturnsVoid();
struct TestDestructor {
  ~TestDestructor() {
    [[clang::musttail]]   // expected-note {{tail call required by 'musttail' attribute here}}
    return ReturnsVoid(); // expected-error {{destructor '~TestDestructor' must not return void expression}}  // expected-error {{cannot perform a tail call from a destructor}}
  }
};

struct ClassWithDestructor { // expected-note {{target destructor is declared here}}
  void TestExplicitDestructorCall() {
    [[clang::musttail]]                  // expected-note {{tail call required by 'musttail' attribute here}}
    return this->~ClassWithDestructor(); // expected-error {{cannot perform a tail call to a destructor}}
  }
};

struct HasNonTrivialCopyConstructor {
  HasNonTrivialCopyConstructor(const HasNonTrivialCopyConstructor &);
};
HasNonTrivialCopyConstructor ReturnsClassByValue();
HasNonTrivialCopyConstructor TestNonElidableCopyConstructor() {
  // This is an elidable constructor, but when it is written explicitly
  // we decline to elide it.
  [[clang::musttail]] return HasNonTrivialCopyConstructor(ReturnsClassByValue()); // expected-error{{'musttail' attribute requires that the return value is the result of a function call}}
}

struct ClassWithConstructor {
  ClassWithConstructor() = default; // expected-note {{target constructor is declared here}}
};
void TestExplicitConstructorCall(ClassWithConstructor a) {
  [[clang::musttail]]                                    // expected-note {{tail call required by 'musttail' attribute here}}
  return a.ClassWithConstructor::ClassWithConstructor(); // expected-error{{cannot perform a tail call to a constructor}}  expected-warning{{explicit constructor calls are a Microsoft extension}}
}

void TestStatementExpression() {
  ({
    HasNonTrivialDestructor foo;               // expected-note {{jump exits scope of variable with non-trivial destructor}}
    [[clang::musttail]] return ReturnsVoid2(); // expected-error {{cannot perform a tail call from this return statement}}
  });
}

struct MyException {};
void TestTryBlock() {
  try {                                        // expected-note {{jump exits try block}}
    [[clang::musttail]] return ReturnsVoid2(); // expected-error {{cannot perform a tail call from this return statement}}
  } catch (MyException &e) {
  }
}

using IntFunctionType = int();
IntFunctionType *ReturnsIntFunction();
long TestRValueFunctionPointer() {
  [[clang::musttail]]            // expected-note {{tail call required by 'musttail' attribute here}}
  return ReturnsIntFunction()(); // expected-error{{cannot perform a tail call to function because its signature is incompatible with the calling function}}  // expected-note{{target function has different return type ('long' expected but has 'int')}}
}

void TestPseudoDestructor() {
  int n;
  using T = int;
  [[clang::musttail]] // expected-note {{tail call required by 'musttail' attribute here}}
  return n.~T();      // expected-error{{cannot perform a tail call to a destructor}}
}

struct StructPMF {
  typedef void (StructPMF::*PMF)();
  static void TestReturnsPMF();
};

StructPMF *St;
StructPMF::PMF ReturnsPMF();
void StructPMF::TestReturnsPMF() {
  [[clang::musttail]]           // expected-note{{tail call required by 'musttail' attribute here}}
  return (St->*ReturnsPMF())(); // expected-error{{static member function cannot perform a tail call to pointer-to-member function}}
}

// These tests are merely verifying that we don't crash with incomplete or
// erroneous ASTs. These cases crashed the compiler in early iterations.

struct TestBadPMF {
  int (TestBadPMF::*pmf)();
  void BadPMF() {
    [[clang::musttail]] return ((*this)->*pmf)(); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'TestBadPMF'}}
  }
};

namespace ns {}
void TestCallNonValue() {
  [[clang::musttail]] return ns; // expected-error {{unexpected namespace name 'ns': expected expression}}
}
