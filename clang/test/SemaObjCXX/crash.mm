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
} // note the missing semicolon

  typedef std::pair<int, int> IntegerPair; // expected-error{{typedef declarator cannot be qualified}} \
// expected-error{{typedef name must be an identifier}} \
// expected-error{{expected ';' after top level declarator}}

@end
