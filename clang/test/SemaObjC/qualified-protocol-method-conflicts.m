// RUN: %clang_cc1  -Woverriding-method-mismatch -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://6191214

@protocol Xint
-(void) setX: (int) arg0; // expected-note {{previous declaration is here}}
+(int) C; // expected-note {{previous declaration is here}}
@end

@protocol Xfloat
-(void) setX: (float) arg0; // expected-note 2 {{previous declaration is here}}
+(float) C;		    // expected-note 2 {{previous declaration is here}}
@end

@interface A <Xint, Xfloat>
@end

@implementation A
-(void) setX: (int) arg0 { } // expected-warning {{conflicting parameter types in declaration of 'setX:': 'float' vs 'int'}}
+(int) C {return 0; } // expected-warning {{conflicting return type in declaration of 'C': 'float' vs 'int'}}
@end

@interface B <Xfloat, Xint>
@end

@implementation B 
-(void) setX: (float) arg0 { } // expected-warning {{conflicting parameter types in declaration of 'setX:': 'int' vs 'float'}}
+ (float) C {return 0.0; } // expected-warning {{conflicting return type in declaration of 'C': 'int' vs 'float'}}
@end

@protocol Xint_float<Xint, Xfloat>
@end

@interface C<Xint_float>
@end

@implementation C
-(void) setX: (int) arg0 { } // expected-warning {{conflicting parameter types in declaration of 'setX:': 'float' vs 'int'}}
+ (int) C {return 0;} // expected-warning {{conflicting return type in declaration of 'C': 'float' vs 'int'}}
@end
