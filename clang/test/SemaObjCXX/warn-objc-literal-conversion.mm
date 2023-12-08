// RUN: %clang_cc1 -fsyntax-only -verify -Wobjc-literal-conversion %s

@class NSString;

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(bool)value;
@end

@interface NSArray
+ (id)arrayWithObjects:(const id [])objects count:(int)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

void char_test() {
  if (@'a') {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void int_test() {
  if (@12) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@-12) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@12LL) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@-12LL) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void float_test() {
  if (@12.55) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@-12.55) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@12.55F) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@-12.55F) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void bool_test() {
  if (@true) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
  if (@false) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void string_test() {
  if (@"asdf") {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void array_test() {
  if (@[ @313, @331, @367, @379 ]) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void dictionary_test() {
  if (@{ @0: @0, @1: @1, @2: @1, @3: @3 }) {}
  // expected-warning@-1{{implicit boolean conversion of Objective-C object literal always evaluates to true}}
}

void objc_bool_test () {
  if (__objc_yes) {}
  if (__objc_no) {}
}
