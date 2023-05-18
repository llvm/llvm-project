// RUN: %clang_cc1 %s -verify -fsyntax-only -fobjc-exceptions

struct some_struct;

@interface NSObject
@end

// Note: NSException is not declared.
void f0(id x) {
  @try {
  } @catch (NSException *x) { // expected-error {{unknown type name 'NSException'}}
  } @catch (struct some_struct x) { // expected-error {{@catch parameter is not a pointer to an interface type}}
  } @catch (int x) { // expected-error {{@catch parameter is not a pointer to an interface type}}
  } @catch (static NSObject *y) { // expected-error {{@catch parameter cannot have storage specifier 'static'}}
  } @catch (...) {
  }
}

