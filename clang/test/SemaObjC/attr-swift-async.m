// RUN: %clang_cc1                 -verify -fsyntax-only -fblocks %s
// RUN: %clang_cc1 -xobjective-c++ -verify -fsyntax-only -fblocks %s

#define SA(...) __attribute__((swift_async(__VA_ARGS__)))

SA(none) int a; // expected-warning{{'swift_async' attribute only applies to functions and Objective-C methods}}

SA(none) void b();

SA(not_swift_private, 0) void c(); // expected-error{{'swift_async' attribute parameter 2 is out of bounds}}
SA(swift_private, 1) void d(); // expected-error{{'swift_async' attribute parameter 2 is out of bounds}}
SA(swift_private, 1) void e(int); // expected-error{{'swift_async' completion handler parameter must have block type returning 'void', type here is 'int'}}
SA(not_swift_private, 1) void f(int (^)()); // expected-error{{'swift_async' completion handler parameter must have block type returning 'void', type here is 'int (^)()'}}
SA(swift_private, 1) void g(void (^)());

SA(none, 1) void h(); // expected-error{{'swift_async' attribute takes one argument}}
SA() void i(); // expected-error{{'swift_async' attribute takes at least 1 argument}}
SA(not_swift_private) void j(); // expected-error{{'swift_async' attribute requires exactly 2 arguments}}
SA(43) void k(); // expected-error{{'swift_async' attribute requires parameter 1 to be an identifier}}
SA(not_a_thing, 0) void l(); // expected-error{{first argument to 'swift_async' must be either 'none', 'swift_private', or 'not_swift_private'}}

@interface TestOnMethods
-(void)m1:(int (^)())callback SA(swift_private, 1); // expected-error{{'swift_async' completion handler parameter must have block type returning 'void', type here is 'int (^)()'}}
-(void)m2:(void (^)())callback SA(swift_private, 0); // expected-error{{'swift_async' attribute parameter 2 is out of bounds}}
-(void)m3:(void (^)())callback SA(swift_private, 2); // expected-error{{'swift_async' attribute parameter 2 is out of bounds}}
-(void)m4 SA(none);
-(void)m5:(int)p handler:(void (^)(int))callback SA(not_swift_private, 2);
@end

#ifdef __cplusplus
struct S {
  SA(none) void mf1();
  SA(swift_private, 2) void mf2(void (^)());
  SA(swift_private, 1) void mf3(void (^)()); // expected-error{{'swift_async' attribute is invalid for the implicit this argument}}
  SA(swift_private, 0) void mf4(void (^)()); // expected-error{{'swift_async' attribute parameter 2 is out of bounds}}
  SA(not_swift_private, 2) void mf5(int (^)()); // expected-error{{'swift_async' completion handler parameter must have block type returning 'void', type here is 'int (^)()'}}
};
#endif
