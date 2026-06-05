// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -Wrelaxed-constant-fold %s

typedef long long LONG_PTR;
typedef long LONG;
#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))
#define FIELD_OFFSET2(type, field) (reinterpret_cast<LONG_PTR>(&(((type *)0)->field)))

struct S {
  int x;
  int y;
} ob;


template<bool B>
struct TplBool {};

TplBool<FIELD_OFFSET(S, y)> tc; // expected-error {{non-type template argument evaluates to 4, which cannot be narrowed to type 'bool'}}
			        // expected-warning@-1 {{folding constant expression involving cast that performs the conversions of a reinterpret_cast is a Microsoft extension}}
constexpr long b = FIELD_OFFSET(S, y); // expected-warning {{folding constant expression involving cast that performs the conversions of a reinterpret_cast is a Microsoft extension}}
constexpr long b2 = FIELD_OFFSET2(S, y); // expected-warning {{folding constant expression involving cast that performs the conversions of a reinterpret_cast is a Microsoft extension}}
constexpr LONG_PTR b3 = (LONG_PTR)&ob; // expected-error {{constexpr variable 'b3' must be initialized by a constant expression}}
				       // expected-note@-1 {{constant expression contains l-value}}
constexpr int* b4 = reinterpret_cast<int*>(&ob); // expected-error {{constexpr variable 'b4' must be initialized by a constant expression}}
						 // expected-note@-1 {{reinterpret_cast is not allowed in a constant expression}}
constexpr LONG_PTR b5 = (42 - FIELD_OFFSET(S, y)) +       // expected-error {{constexpr variable 'b5' must be initialized by a constant expression}}
                (8 + reinterpret_cast<LONG_PTR>(&ob));    // expected-note {{constant expression contains l-value}}
constexpr LONG_PTR b6 = -reinterpret_cast<LONG_PTR>(&ob); // expected-error {{constexpr variable 'b6' must be initialized by a constant expression}}
							  // expected-note@-1 {{constant expression contains l-value}}
constexpr LONG_PTR b7[2] = { FIELD_OFFSET(S, y), (LONG_PTR)&ob }; // expected-error {{constexpr variable 'b7' must be initialized by a constant expression}}
		  					          // expected-note@-1 {{constant expression contains l-value}}
constexpr LONG_PTR b8  = (LONG_PTR)((char*)1 + FIELD_OFFSET(S, y)); // expected-error {{constexpr variable 'b8' must be initialized by a constant expression}}
								    // expected-note@-1 {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}
constexpr LONG_PTR b9  = (LONG_PTR)(FIELD_OFFSET(S, y) / 0); // expected-error {{constexpr variable 'b9' must be initialized by a constant expression}}
							     // expected-note@-1 {{division by zero}}
