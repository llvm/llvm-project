// RUN:  %clang_cc1 -std=c++2a -fconcepts-ts -verify %s
// Support parsing of concepts

concept X;  //expected-error {{must be a template}}

template<typename T> concept C1 = true; //expected-note{{declared here}}  <-- previous declaration

template<typename T> concept C1 = true; // expected-error {{redefinition of 'C1'}}

template<concept T> concept D1 = true; // expected-error {{'concept' can only appear in namespace scope}} \
                                          expected-error {{expected template parameter}}

template<> concept X = true; // expected-error {{cannot be explicitly specialized}}

namespace ns1 {
template<> concept D1 = true;  // expected-error {{cannot be explicitly specialized}}
template<typename T> const concept C1 = true; //expected-error{{cannot combine with}}
namespace ns12 {
template<typename T> decltype(T{}) concept C2 = true; //expected-error{{cannot combine with}}
template<typename T> bool concept C3 = true; //expected-error{{cannot combine with}}
template<typename T> unsigned concept C4 = true; //expected-error{{cannot combine with}}
template<typename T> short concept C5 = true; //expected-error{{cannot combine with}}
template<typename T> typedef concept C6 = true; //expected-error{{cannot combine with}}
}
template<class> concept 
                        const  //expected-error{{expected concept name}}
						      C2 = true; 
							  
void f() {
	concept X; //expected-error{{'concept' can only appear in namespace scope}}
}
template<concept X,  //expected-error{{'concept' can only appear in namespace scope}} \
                                          expected-error {{expected template parameter}}
				  int J> void f();
}

template<class T>
concept [[test]] X2 [[test2]] = T::value; //expected-error2{{attribute list cannot appear here}}

namespace ns2 {
template<class T>
concept [[test]] X2 [[test2]] = T::value; //expected-error2{{attribute list cannot appear here}}
	
}

namespace ns3 {
   template<typename T> concept C1 = true; 

  namespace ns1 {
	using ns3::C1; //expected-note{{declared here}}
	template<typename T> concept C1 = true; // expected-error {{redefinition of 'C1'}}
  }

}

// TODO:
// template<typename T> concept C2 = 0.f; // expected error {{constraint expression must be 'bool'}}

struct S1 {
  template<typename T> concept C1 = true; // expected-error {{can only appear in namespace scope}}
};

template<typename A>
template<typename B>
concept C4 = true; // expected-error {{extraneous template parameter list in concept definition}}

template<typename T> concept C5 = true; // expected-note {{previous}} expected-note {{previous}}
int C5; // expected-error {{redefinition}}
struct C5 {}; // expected-error {{redefinition}}

struct C6 {}; //expected-note{{declared here}}  <-- previous declaration
template<typename T> concept C6 = true; // expected-error {{redefinition of 'C6' as different kind of symbol}}

namespace thing {};

template<typename T> concept thing::C7 = true;  // expected-error {{concepts must be defined in their own namespace}}


namespace ns5 {
}

// TODO: Add test to prevent explicit specialization, partial specialization
// and explicit instantiation of concepts.
