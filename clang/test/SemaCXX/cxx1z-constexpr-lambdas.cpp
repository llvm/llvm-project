// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks %s
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s 
// RUN: %clang_cc1 -std=c++14 -verify -fsyntax-only -fblocks %s -DCPP14_AND_EARLIER


namespace test_lambda_is_literal {
#ifdef CPP14_AND_EARLIER
//expected-error@+4{{not a literal type}}
//expected-note@+2{{not an aggregate and has no constexpr constructors}}
#endif
auto L = [] { };
constexpr int foo(decltype(L) l) { return 0; }

}

#ifndef CPP14_AND_EARLIER
namespace test_constexpr_checking {

namespace ns1 {
  struct NonLit { ~NonLit(); };  //expected-note{{not literal}}
  auto L = [](NonLit NL) constexpr { }; //expected-error{{not a literal type}}
} // end ns1

namespace ns2 {
  auto L = [](int I) constexpr { asm("non-constexpr");  }; //expected-error{{not allowed in constexpr function}}
} // end ns1

} // end ns test_constexpr_checking

namespace test_constexpr_call {

namespace ns1 {
  auto L = [](int I) { return I; };
  static_assert(L(3) == 3);
} // end ns1
namespace ns2 {
  auto L = [](auto a) { return a; };
  static_assert(L(3) == 3);
  static_assert(L(3.14) == 3.14);
}
namespace ns3 {
  auto L = [](auto a) { asm("non-constexpr"); return a; }; //expected-note{{declared here}}
  constexpr int I =  //expected-error{{must be initialized by a constant expression}}
      L(3); //expected-note{{non-constexpr function}}
} 

} // end ns test_constexpr_call

namespace test_captureless_lambda {
void f() {
  const char c = 'c';
  auto L = [] { return c; };
  constexpr char C = L();
}
  
void f(char c) { //expected-note{{declared here}}
  auto L = [] { return c; }; //expected-error{{cannot be implicitly captured}} expected-note{{lambda expression begins here}}
  int I = L();
}

}

namespace test_conversion_function_for_non_capturing_lambdas {

namespace ns1 {
auto L = [](int i) { return i; };
constexpr int (*fpi)(int) = L;
static_assert(fpi(3) == 3);
auto GL = [](auto a) { return a; };

constexpr char (*fp2)(char) = GL;
constexpr double (*fp3)(double) = GL;
constexpr const char* (*fp4)(const char*) = GL;
static_assert(fp2('3') == '3');
static_assert(fp3(3.14) == 3.14);
constexpr const char *Str = "abc";
static_assert(fp4(Str) == Str);

auto NCL = [](int i) { static int j; return j; }; //expected-note{{declared here}}
constexpr int (*fp5)(int) = NCL;
constexpr int I =  //expected-error{{must be initialized by a constant expression}}
                  fp5(5); //expected-note{{non-constexpr function}} 

namespace test_dont_always_instantiate_constexpr_templates {

auto explicit_return_type = [](auto x) -> int { return x.get(); };
decltype(explicit_return_type(0)) c;  // OK

auto deduced_return_type = [](auto x) { return x.get(); }; //expected-error{{not a structure or union}}
decltype(deduced_return_type(0)) d;  //expected-note{{requested here}}


  
} // end ns test_dont_always_instantiate_constexpr_templates
} // end ns1

} // end ns test_conversion_function_for_non_capturing_lambdas

namespace test_lambda_is_cce {
namespace ns1_simple_lambda {

namespace ns0 {
constexpr int I = [](auto a) { return a; }(10);

static_assert(I == 10); 
static_assert(10 == [](auto a) { return a; }(10));
static_assert(3.14 == [](auto a) { return a; }(3.14));

} //end ns0

namespace ns1 {
constexpr auto f(int i) {
  double d = 3.14;
  auto L = [=](auto a) { 
    int Isz = sizeof(i);
    return sizeof(i) + sizeof(a) + sizeof(d); 
  };
  int I = L("abc") + L(nullptr);
  return L;
}
constexpr auto L = f(3);
constexpr auto M =  L("abc") + L(nullptr);

static_assert(M == sizeof(int) * 2 + sizeof(double) * 2 + sizeof(nullptr) + sizeof(const char*));

} // end ns1

namespace ns2 {
constexpr auto f(int i) {
  auto L = [](auto a) { return a + a; };
  return L;
}
constexpr auto L = f(3);
constexpr int I = L(6);
static_assert(I == 12);
} // end ns2

namespace contained_lambdas_call_operator_is_not_constexpr {
constexpr auto f(int i) {
  double d = 3.14;
  auto L = [=](auto a) { //expected-note{{declared here}}
    int Isz = sizeof(i);
    asm("hello");
    return sizeof(i) + sizeof(a) + sizeof(d); 
  };
  return L;
}

constexpr auto L = f(3);

constexpr auto M =  // expected-error{{must be initialized by}} 
    L("abc"); //expected-note{{non-constexpr function}}

} // end ns contained_lambdas_call_operator_is_not_constexpr



} // end ns1_simple_lambda

namespace ns1_unimplemented {
namespace ns1_captures {
constexpr auto f(int i) {
  double d = 3.14;
  auto L = [=](auto a) { //expected-note{{coming soon}}
    int Isz = i + d;
    return sizeof(i) + sizeof(a) + sizeof(d); 
  };
  return L;
}
constexpr auto M = f(3);  //expected-error{{constant expression}} expected-note{{in call to}}
} // end ns1_captures
} // end ns1_unimplemented 

} // end ns test_lambda_is_cce

#endif // ndef CPP14_AND_EARLIER
