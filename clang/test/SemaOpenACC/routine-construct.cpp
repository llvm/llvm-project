// RUN: %clang_cc1 %s -fopenacc -verify

// expected-error@+1{{use of undeclared identifier 'UnnamedYet'}}
#pragma acc routine(UnnamedYet)
void UnnamedYet();
// expected-error@+1{{use of undeclared identifier 'Invalid'}}
#pragma acc routine(Invalid)

// Fine, since these are the same function.
void SameFunc();
#pragma acc routine(SameFunc)
void SameFunc();

void NoMagicStatic() {
  static int F = 1;
}
// expected-error@-2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine(NoMagicStatic)

void NoMagicStatic2();
// expected-error@+4{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine(NoMagicStatic2)
void NoMagicStatic2() {
  static int F = 1;
}

void HasMagicStaticLambda() {
  auto MSLambda = []() {
    static int I = 5;
  };
// expected-error@-2{{function static variables are not permitted in functions to which an OpenACC 'routine' directive applies}}
// expected-note@+1{{'routine' construct is here}}
#pragma acc routine (MSLambda)
}

auto Lambda = [](){};
#pragma acc routine(Lambda)
auto GenLambda = [](auto){};
// expected-error@+1{{OpenACC routine name 'GenLambda' names a set of overloads}}
#pragma acc routine(GenLambda)
// Variable?
int Variable;
// Plain function
int function();

#pragma acc routine (function)
// expected-error@+1{{OpenACC routine name 'Variable' does not name a function}}
#pragma acc routine (Variable)

// Var template?
template<typename T>
T VarTempl = 0;
// expected-error@+2{{use of variable template 'VarTempl' requires template arguments}}
// expected-note@-2{{template is declared here}}
#pragma acc routine (VarTempl)
// expected-error@+1{{OpenACC routine name 'VarTempl<int>' does not name a function}}
#pragma acc routine (VarTempl<int>)

// Function in NS
namespace NS {
  int NSFunc();
auto Lambda = [](){};
}
#pragma acc routine(NS::NSFunc)
#pragma acc routine(NS::Lambda)

// Ambiguous Function
int ambig_func();
int ambig_func(int);
// expected-error@+1{{OpenACC routine name 'ambig_func' names a set of overloads}}
#pragma acc routine (ambig_func)

// Ambiguous in NS
namespace NS {
int ambig_func();
int ambig_func(int);
}
// expected-error@+1{{OpenACC routine name 'NS::ambig_func' names a set of overloads}}
#pragma acc routine (NS::ambig_func)

// function template
template<typename T, typename U>
void templ_func();

// expected-error@+1{{OpenACC routine name 'templ_func' names a set of overloads}}
#pragma acc routine(templ_func)
// expected-error@+1{{OpenACC routine name 'templ_func<int>' names a set of overloads}}
#pragma acc routine(templ_func<int>)
// expected-error@+1{{OpenACC routine name 'templ_func<int, float>' names a set of overloads}}
#pragma acc routine(templ_func<int, float>)

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

#pragma acc routine(S::MemFunc)
#pragma acc routine(S::StaticMemFunc)
#pragma acc routine(S::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(S::Field)
};

#pragma acc routine(S::MemFunc)
#pragma acc routine(S::StaticMemFunc)
#pragma acc routine(S::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(S::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(S::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(S::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(S::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(S::Field)

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

#pragma acc routine(DepS::MemFunc)
#pragma acc routine(DepS::StaticMemFunc)
#pragma acc routine(DepS::Lambda)
#pragma acc routine(DepS::LambdaBroken)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::Field' does not name a function}}
#pragma acc routine(DepS::Field)

#pragma acc routine(DepS<T>::MemFunc)
#pragma acc routine(DepS<T>::StaticMemFunc)
#pragma acc routine(DepS<T>::Lambda)
#pragma acc routine(DepS<T>::LambdaBroken)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<T>::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<T>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<T>::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS<T>::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'DepS<T>::Field' does not name a function}}
#pragma acc routine(DepS<T>::Field)
};

void Inst() {
  DepS<int> S; // #DEPSInst
}


//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::Lambda)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::MemFunc)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::StaticMemFunc)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFunc)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplStaticMemFunc)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFunc<int>)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplStaticMemFunc<int>)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::MemFuncAmbig)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFuncAmbig)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::TemplMemFuncAmbig<int>)
//expected-error@+2{{use of class template 'DepS' requires template arguments}}
// expected-note@#DEPS{{template is declared here}}
#pragma acc routine(DepS::Field)

//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::Lambda)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::MemFunc)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::StaticMemFunc)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFunc)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFunc<int>)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplStaticMemFunc<int>)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::MemFuncAmbig)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::TemplMemFuncAmbig<int>)
//expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(DepS<T>::Field)

#pragma acc routine(DepS<int>::Lambda)
#pragma acc routine(DepS<int>::LambdaBroken)
#pragma acc routine(DepS<int>::MemFunc)
#pragma acc routine(DepS<int>::StaticMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFunc' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'DepS<int>::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<int>::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'DepS<int>::TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(DepS<int>::TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'DepS<int>::Field' does not name a function}}
#pragma acc routine(DepS<int>::Field)

template<typename T>
void TemplFunc() {
#pragma acc routine(T::MemFunc)
#pragma acc routine(T::StaticMemFunc)
#pragma acc routine(T::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field)
}

template <typename T>
struct DepRefersToT {
#pragma acc routine(T::MemFunc)
#pragma acc routine(T::StaticMemFunc)
#pragma acc routine(T::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field)

  void MemFunc() {
#pragma acc routine(T::MemFunc)
#pragma acc routine(T::StaticMemFunc)
#pragma acc routine(T::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field)
  }

  template<typename U>
  void TemplMemFunc() {
#pragma acc routine(T::MemFunc)
#pragma acc routine(T::StaticMemFunc)
#pragma acc routine(T::Lambda)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplMemFunc)
// expected-error@+1{{OpenACC routine name 'S::TemplStaticMemFunc' names a set of overloads}}
#pragma acc routine(T::TemplStaticMemFunc)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::template TemplStaticMemFunc<int>' names a set of overloads}}
#pragma acc routine(T::template TemplStaticMemFunc<int>)
// expected-error@+1{{OpenACC routine name 'S::MemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::MemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::TemplMemFuncAmbig' names a set of overloads}}
#pragma acc routine(T::TemplMemFuncAmbig)
// expected-error@+1{{OpenACC routine name 'S::template TemplMemFuncAmbig<int>' names a set of overloads}}
#pragma acc routine(T::template TemplMemFuncAmbig<int>)
// expected-error@+1{{OpenACC routine name 'S::Field' does not name a function}}
#pragma acc routine(T::Field)
  }

};

void inst() {
  TemplFunc<S>(); // expected-note{{in instantiation of}}
  DepRefersToT<S> s; // expected-note{{in instantiation of}}
  s.MemFunc(); // expected-note{{in instantiation of}}
  s.TemplMemFunc<S>(); // expected-note{{in instantiation of}}
}
