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
