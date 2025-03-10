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

void HasMagicStaticLambda() {
  auto MSLambda = []() {
    static int I = 5;
  };
// expected-error@-2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine (MSLambda) seq
}

auto Lambda = [](){};
#pragma acc routine(Lambda) seq
auto GenLambda = [](auto){};
// expected-error@+1{{OpenACC routine name 'GenLambda' names a set of overloads}}
#pragma acc routine(GenLambda) seq
// Variable?
int Variable;
// Plain function
int function();

#pragma acc routine (function) seq
// expected-error@+1{{OpenACC routine name 'Variable' does not name a function}}
#pragma acc routine (Variable) seq

// Var template?
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

// expected-error@+1{{OpenACC routine name 'templ_func' names a set of overloads}}
#pragma acc routine(templ_func) seq
// expected-error@+1{{OpenACC routine name 'templ_func<int>' names a set of overloads}}
#pragma acc routine(templ_func<int>) seq
// expected-error@+1{{OpenACC routine name 'templ_func<int, float>' names a set of overloads}}
#pragma acc routine(templ_func<int, float>) seq

struct S {
  void MemFunc();
  static void StaticMemFunc();
  template<typename U>
  void TemplMemFunc();
  template<typename U>
  static void TemplStaticMemFunc();

  void MemFuncAmbig();
  void MemFuncAmbig(int);
  template<typename T>
  void TemplMemFuncAmbig();
  template<typename T>
  void TemplMemFuncAmbig(int);

  int Field;

  constexpr static auto Lambda = [](){};

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
};

void Inst() {
  DepS<int> S; // #DEPSInst
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
  }

};

void inst() {
  TemplFunc<S>(); // expected-note{{in instantiation of}}
  DepRefersToT<S> s; // expected-note{{in instantiation of}}
  s.MemFunc(); // expected-note{{in instantiation of}}
  s.TemplMemFunc<S>(); // expected-note{{in instantiation of}}
}
