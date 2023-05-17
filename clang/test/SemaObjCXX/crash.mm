// RUN: %clang_cc1 -fsyntax-only %s -verify 
// RUN: %clang_cc1 -fsyntax-only -std=c++98 %s -verify 
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify 

// <rdar://problem/11286701>
namespace std {
  template<typename T, typename U> class pair;
}

@interface NSObject
@end

@interface Test : NSObject
@end

@implementation Test

struct EvilStruct {
} // expected-error {{expected ';' after struct}}

  typedef std::pair<int, int> IntegerPair;

template<typename...Ts> void f(Ts); // expected-error {{unexpanded}}
#if __cplusplus <= 199711L // C++03 or earlier modes
// expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif
@end

// rdar://20560175

struct OuterType {
  typedef int InnerType;
};

namespace ns {
  typedef int InnerType;
};

@protocol InvalidProperties

@property (nonatomic) (OuterType::InnerType) invalidTypeParens;
// expected-error@-1 {{type name requires a specifier or qualifier}}
// expected-error@-2 {{property requires fields to be named}}
// expected-error@-3 {{expected ';' at end of declaration list}}
// expected-error@-4 {{a type specifier is required for all declarations}}
// expected-error@-5 {{cannot declare variable inside @interface or @protocol}}

@property (nonatomic) (ns::InnerType) invalidTypeParens2;
// expected-error@-1 {{type name requires a specifier or qualifier}}
// expected-error@-2 {{property requires fields to be named}}
// expected-error@-3 {{expected ';' at end of declaration list}}
// expected-error@-4 {{a type specifier is required for all declarations}}
// expected-error@-5 {{cannot declare variable inside @interface or @protocol}}

@property (nonatomic) int OuterType::InnerType; // expected-error {{property requires fields to be named}}

@property (nonatomic) int OuterType::InnerType foo; // expected-error {{property requires fields to be named}}
// expected-error@-1 {{expected ';' at end of declaration list}}
// expected-error@-2 {{a type specifier is required for all declarations}}
// expected-error@-3 {{cannot declare variable inside @interface or @protocol}}

@end

// This used to crash.
@protocol Property0;
@protocol Property0;
id<Property0> x;
@protocol Property0;
id<Property0> y;
id<Property0> z = true ? x : y;
