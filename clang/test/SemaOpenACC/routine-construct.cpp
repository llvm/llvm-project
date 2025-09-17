// RUN: %clang_cc1 %s -fopenacc -verify

// expected-error@+1{{use of undeclared identifier 'UnnamedYet'}}
#pragma acc routine(UnnamedYet) seq
void UnnamedYet();
// expected-error@+1{{use of undeclared identifier 'Invalid'}}
#pragma acc routine(Invalid) seq

// Fine, since these are the same function.
void SameFunc();
#pragma acc routine(SameFunc) seq
void SameFunc();

namespace NS {
void DifferentFunc();
};
// expected-warning@+2{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@-3{{'DifferentFunc' declared here}}
#pragma acc routine(NS::DifferentFunc) seq
void DifferentFunc();

void NoMagicStatic() {
  static int F = 1;
}
// expected-error@-2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine(NoMagicStatic) seq

void NoMagicStatic2();
// expected-error@+4{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine(NoMagicStatic2) seq
void NoMagicStatic2() {
  static int F = 1;
}

#pragma acc routine seq
void NoMagicStatic3() {
  // expected-error@+2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
  // expected-note@-3{{'routine' construct is here}}
  static int F = 1;
}

#pragma acc routine seq
void NoMagicStatic4();
void NoMagicStatic4() {
  // expected-error@+2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
  // expected-note@-4{{'routine' construct is here}}
  static int F = 1;
}

void HasMagicStaticLambda() {
  auto MSLambda = []() {
    static int I = 5;
  };
// expected-error@-2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine (MSLambda) seq

// expected-error@+4{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine seq
  auto MSLambda2 = []() {
    static int I = 5;
  };

// Properly handle error recovery.
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  auto MSLambda2 = [](auto) {
    // expected-error@-1{{redefinition of 'MSLambda2'}}
    // expected-note@-9{{previous definition is here}}
    static int I = 5;
  };
// expected-error@+4{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine seq
  auto MSLambda3 = [](auto) {
    static int I = 5;
  };
}

auto Lambda = [](){};
#pragma acc routine(Lambda) seq

#pragma acc routine seq
auto Lambda2 = [](){};
auto GenLambda = [](auto){};
// expected-error@+1{{OpenACC routine name 'GenLambda' names a set of overloads}}
#pragma acc routine(GenLambda) seq

#pragma acc routine seq
auto GenLambda2 = [](auto){};

// Variable?
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
int Variable;
// Plain function
#pragma acc routine seq
int function();

#pragma acc routine (function) seq
// expected-error@+1{{OpenACC routine name 'Variable' does not name a function}}
#pragma acc routine (Variable) seq

// Var template?
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
template<typename T>
T VarTempl = 0;
// expected-error@+2{{use of variable template 'VarTempl' requires template arguments}}
// expected-note@-2{{template is declared here}}
#pragma acc routine (VarTempl) seq
// expected-error@+1{{OpenACC routine name 'VarTempl<int>' does not name a function}}
#pragma acc routine (VarTempl<int>) seq

// Function in NS
namespace NS {
  int NSFunc();
auto Lambda = [](){};
}
#pragma acc routine(NS::NSFunc) seq
#pragma acc routine(NS::Lambda) seq

// Ambiguous Function
int ambig_func();
int ambig_func(int);
// expected-error@+1{{OpenACC routine name 'ambig_func' names a set of overloads}}
#pragma acc routine (ambig_func) seq

#pragma acc routine seq
int ambig_func2();
#pragma acc routine seq
int ambig_func2(int);

// Ambiguous in NS
namespace NS {
int ambig_func();
int ambig_func(int);
}
// expected-error@+1{{OpenACC routine name 'NS::ambig_func' names a set of overloads}}
#pragma acc routine (NS::ambig_func) seq

// function template
template<typename T, typename U>
void templ_func();
#pragma acc routine seq
template<typename T, typename U>
void templ_func2();

// expected-error@+1{{OpenACC routine name 'templ_func' names a set of overloads}}
#pragma acc routine(templ_func) seq
// expected-error@+1{{OpenACC routine name 'templ_func<int>' names a set of overloads}}
#pragma acc routine(templ_func<int>) seq
// expected-error@+1{{OpenACC routine name 'templ_func<int, float>' names a set of overloads}}
#pragma acc routine(templ_func<int, float>) seq

struct S {
  void MemFunc();
#pragma acc routine seq
  void MemFunc2();
  static void StaticMemFunc();
#pragma acc routine seq
  static void StaticMemFunc2();
  template<typename U>
  void TemplMemFunc();
#pragma acc routine seq
  template<typename U>
  void TemplMemFunc2();
  template<typename U>
  static void TemplStaticMemFunc();
#pragma acc routine seq
  template<typename U>
  static void TemplStaticMemFunc2();

  void MemFuncAmbig();
  void MemFuncAmbig(int);
  template<typename T>
  void TemplMemFuncAmbig();
  template<typename T>
  void TemplMemFuncAmbig(int);

  int Field;

  constexpr static auto Lambda = [](){};
#pragma acc routine seq
  constexpr static auto Lambda2 = [](){};
#pragma acc routine seq
  constexpr static auto Lambda3 = [](auto){};

#pragma acc routine(S::MemFunc) seq
#pragma acc routine(S::StaticMemFunc) seq
#pragma acc routine(S::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(S::Field) seq
};

#pragma acc routine(S::MemFunc) seq
#pragma acc routine(S::StaticMemFunc) seq
#pragma acc routine(S::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(S::Field) seq

constexpr auto getLambda() {
  return [](){};
}
template<typename T>
constexpr auto getTemplLambda() {
  return [](T){};
}
constexpr auto getDepLambda() {
  return [](auto){};
}
template<typename T>
constexpr auto getTemplDepLambda() {
  return [](auto){};
}

template<typename T>
struct DepS { // #DEPS
  void MemFunc();
  static void StaticMemFunc();

  template<typename U>
  void TemplMemFunc();
  template<typename U>
  static void TemplStaticMemFunc();

  void MemFuncAmbig();
  void MemFuncAmbig(int);
  template<typename U>
  void TemplMemFuncAmbig();
  template<typename U>
  void TemplMemFuncAmbig(int);

  int Field;
  constexpr static auto Lambda = [](){};
  // expected-error@+2{{non-const static data member must be initialized out of line}}
  // expected-note@#DEPSInst{{in instantiation of template class}}
  static auto LambdaBroken = [](){};

#pragma acc routine seq
  constexpr static auto LambdaKinda = getLambda();
  // FIXME: We can't really handle this/things like this, see comment in
  // SemaOpenACC.cpp's LegalizeNextParsedDecl.
  // expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  constexpr static auto LambdaKinda2 = getTemplLambda<T>();
#pragma acc routine seq
  constexpr static auto DepLambdaKinda = getDepLambda();
  // expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  constexpr static auto DepLambdaKinda2 = getTemplDepLambda<T>();

// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  constexpr static auto Bad = T{};

#pragma acc routine seq
  constexpr static auto LambdaHasMagicStatic = []() {
  // expected-error@+2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
  // expected-note@-3{{'routine' construct is here}}
    static int F = 1;
  };

  void HasMagicStatic() {
    static int F = 1; // #HasMagicStaticFunc
  }
  void HasMagicStatic2() {
    static int F = 1; // #HasMagicStaticFunc2
  }

#pragma acc routine(DepS::MemFunc) seq
#pragma acc routine(DepS::StaticMemFunc) seq
#pragma acc routine(DepS::Lambda) seq
#pragma acc routine(DepS::LambdaBroken) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::Field' does not name a function}}
#pragma acc routine(DepS::Field) seq

#pragma acc routine(DepS<T>::MemFunc) seq
#pragma acc routine(DepS<T>::StaticMemFunc) seq
#pragma acc routine(DepS<T>::Lambda) seq
#pragma acc routine(DepS<T>::LambdaBroken) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<T>::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<T>::Field' does not name a function}}
#pragma acc routine(DepS<T>::Field) seq

// FIXME: We could do better about suppressing this double diagnostic, but we
// don't want to invalidate the vardecl for openacc, so we don't have a good
// way to do this in the AST.
// expected-error@#HasMagicStaticFunc 2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+2 2{{'routine' construct is here}}
// expected-note@+1{{in instantiation of member function}}}
#pragma acc routine(DepS<T>::HasMagicStatic) seq
};

template<typename T>
void DepF() {
#pragma acc routine seq
  auto LambdaKinda = getLambda();
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  auto LambdaKinda2 = getTemplLambda<T>();
#pragma acc routine seq
  auto DepLambdaKinda = getDepLambda();
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  auto DepLambdaKinda2 = getTemplDepLambda<T>();

// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
  constexpr static auto Bad = T{};
}

void Inst() {
  DepS<int> S; // #DEPSInst
  S.HasMagicStatic2();
  DepF<int>(); // #DEPFInst
}


//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::Lambda) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::MemFunc) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::StaticMemFunc) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFunc) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplStaticMemFunc) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFunc<int>) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplStaticMemFunc<int>) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::MemFuncAmbig) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFuncAmbig) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFuncAmbig<int>) seq
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::Field) seq

//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::Lambda) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::MemFunc) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::StaticMemFunc) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFunc) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFunc<int>) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc<int>) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::MemFuncAmbig) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig<int>) seq
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::Field) seq

#pragma acc routine(DepS<int>::Lambda) seq
#pragma acc routine(DepS<int>::LambdaBroken) seq
#pragma acc routine(DepS<int>::MemFunc) seq
#pragma acc routine(DepS<int>::StaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<int>::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'DepS<int>::Field' does not name a function}}
#pragma acc routine(DepS<int>::Field) seq

// expected-error@#HasMagicStaticFunc2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+2{{'routine' construct is here}}
// expected-note@+1{{in instantiation of member function}}}
#pragma acc routine(DepS<int>::HasMagicStatic2) seq

template<typename T>
void TemplFunc() {
#pragma acc routine(T::MemFunc) seq
#pragma acc routine(T::StaticMemFunc) seq
#pragma acc routine(T::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field) seq
}

template <typename T>
struct DepRefersToT {
#pragma acc routine(T::MemFunc) seq
#pragma acc routine(T::StaticMemFunc) seq
#pragma acc routine(T::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field) seq

  void MemFunc() {
#pragma acc routine(T::MemFunc) seq
#pragma acc routine(T::StaticMemFunc) seq
#pragma acc routine(T::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field) seq
  }

  template<typename U>
  void TemplMemFunc() {
#pragma acc routine(T::MemFunc) seq
#pragma acc routine(T::StaticMemFunc) seq
#pragma acc routine(T::Lambda) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>) seq
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig) seq
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>) seq
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field) seq

// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
   auto L = getTemplLambda<U>();
// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq
   auto L2 = getTemplDepLambda<U>();
  }

};

void inst() {
  TemplFunc<S>(); // expected-note{{in instantiation of}}
  DepRefersToT<S> s; // expected-note{{in instantiation of}}
  s.MemFunc(); // expected-note{{in instantiation of}}
  s.TemplMemFunc<S>(); // expected-note{{in instantiation of}}
}

// A.3.4 tests:

void DiffFuncs(); // #GLOBALDIFFFUNCS
namespace NS {
// expected-warning@+2{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@#GLOBALDIFFFUNCS{{'DiffFuncs' declared here}}
#pragma acc routine(DiffFuncs) seq
void DiffFuncs();
}

void has_diff_func() {
// expected-warning@+2{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@#GLOBALDIFFFUNCS{{'DiffFuncs' declared here}}
#pragma acc routine(DiffFuncs) seq
auto DiffFuncs = [](){};
}

template<typename T>
void has_diff_func_templ() {
// expected-warning@+3{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@#GLOBALDIFFFUNCS{{'DiffFuncs' declared here}}
// expected-note@#HDFT_INST{{in instantiation of function template specialization}}
#pragma acc routine(DiffFuncs) seq
auto DiffFuncs = [](){};
}

void inst_diff() {
  has_diff_func_templ<int>();// #HDFT_INST
}

struct SDiff {
// expected-warning@+2{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@#GLOBALDIFFFUNCS{{'DiffFuncs' declared here}}
#pragma acc routine(DiffFuncs) seq
  void DiffFuncs();
};
template<typename T>
struct TemplSDiff {
// expected-warning@+2{{OpenACC 'routine' directive with a name refers to a function with the same name as the function on the following line; this may be unintended}}
// expected-note@#GLOBALDIFFFUNCS{{'DiffFuncs' declared here}}
#pragma acc routine(DiffFuncs) seq
  void DiffFuncs();
};

struct SOperator {
#pragma acc routine(DiffFuncs) seq
  bool operator==(const S&);
};

namespace NS2 {
  // Shouldn't diagnose.
#pragma acc routine(DiffFuncs) seq
#pragma acc routine seq
  void DiffFuncs();
};

