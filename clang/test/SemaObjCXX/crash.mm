// RUN: %clang_cc1 -fsyntax-only %s -verify 

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

template<typename...Ts> void f(Ts); // expected-error {{unexpanded}} expected-warning {{extension}}

@end
