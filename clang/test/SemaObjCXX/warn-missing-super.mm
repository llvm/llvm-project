// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// PR13401

__attribute((objc_root_class)) @interface NSObject
@end

@interface Dummy : NSObject
@end

template<typename T> struct shared_ptr {
  constexpr shared_ptr() {}
};

@implementation Dummy
- (void)dealloc
{
	constexpr shared_ptr<int> dummy;
} // expected-warning {{method possibly missing a [super dealloc] call}}
@end
