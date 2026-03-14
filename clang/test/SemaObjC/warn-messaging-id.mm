// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class -Wobjc-messaging-id %s

@interface CallMeMaybe

- (void)doThing:(int)intThing;

@property int thing;

@end

template<typename T>
void instantiate(const T &x) {
  [x setThing: 22]; // expected-warning {{messaging unqualified id}}
}

void fn() {
  id myObject;
  [myObject doThing: 10]; // expected-warning {{messaging unqualified id}}
  [myObject setThing: 11]; // expected-warning {{messaging unqualified id}}
  instantiate(myObject); // expected-note {{in instantiation}}
}
