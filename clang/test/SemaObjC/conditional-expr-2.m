// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface A
@end
@interface B
@end

void f0(int cond, A *a, B *b) {
  // Ensure that we can still send a message to result of incompatible
  // conditional expression.
  [ (cond ? a : b) test ]; // expected-warning{{incompatible operand types ('A *' and 'B *')}} expected-warning {{method '-test' not found}}
}

@interface NSKey @end
@interface KeySub : NSKey @end

@interface UpdatesList @end

void foo (int i, NSKey *NSKeyValueCoding_NullValue, UpdatesList *nukedUpdatesList)
{
  id obj;
  NSKey *key;
  KeySub *keysub;

  obj = i ? NSKeyValueCoding_NullValue : nukedUpdatesList; // expected-warning{{incompatible operand types ('NSKey *' and 'UpdatesList *')}}
  key = i ? NSKeyValueCoding_NullValue : nukedUpdatesList; // expected-warning{{incompatible operand types ('NSKey *' and 'UpdatesList *')}}
  key = i ? NSKeyValueCoding_NullValue : keysub;
  keysub = i ? NSKeyValueCoding_NullValue : keysub; // expected-warning{{incompatible pointer types assigning to 'KeySub *' from 'NSKey *'}}
}
