// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s

void abort(void);

@interface Subclass
+ (int)magicNumber;
+ (void)setMagicNumber:(int)value;
+ (void)setFakeSetterNumber:(int)value;
@end

@implementation Subclass
int _magicNumber = 0;
+ (int)magicNumber {
  return _magicNumber;
}

+ (void)setMagicNumber:(int)value {
  _magicNumber = value;
}

+ (void)setFakeSetterNumber:(int)value {
  _magicNumber = value;
}

+ (void) classMeth
{
	self.magicNumber = 10;
	if (self.magicNumber != 10)
	  abort ();
}
@end

int main (void) {
  
  int a;
  Subclass.magicNumber = 2 /*[Subclass setMagicNumber:2]*/;
  if (Subclass.magicNumber != 0)
    abort ();
  if (Subclass.magicNumber != 2)
    abort ();
  Subclass.magicNumber += 3;
  if (Subclass.magicNumber != 5)
    abort ();
  Subclass.magicNumber -= 5;
  if (Subclass.magicNumber != 0)
    abort ();
  /* We only have a setter in the following case. */
  Subclass.fakeSetterNumber = 123;

  /* We read it using the other getter. */
  if (Subclass.magicNumber != 123)
   abort ();
  Subclass.fakeSetterNumber = Subclass.magicNumber;
  if (Subclass.magicNumber != 123)
   abort ();

  Subclass.fakeSetterNumberX = 123; // expected-error{{property 'fakeSetterNumberX' not found on object of type 'Subclass'}}

  /* Test class methods using the new syntax. */
  [Subclass classMeth];
  return 0;
}
