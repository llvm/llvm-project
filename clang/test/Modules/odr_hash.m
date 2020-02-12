// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/second.h -fblocks -fobjc-arc

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/Inputs -verify %s -fblocks -fobjc-arc

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST) || defined(SECOND)
@protocol NSObject
@end

__attribute__((objc_root_class))
@interface NSObject
@end

@protocol P1
@end

@protocol P2
@end

@protocol UP1;
@protocol UP2;
@protocol UP3;

@interface I1
@end

@interface I2 : I1
@end

@interface Interface1 <T : I1 *> {
@public
  T x;
}
@end

@interface Interface2 <T : I1 *>
@end

@interface Interface3 <T : I1 *>
@end

@interface EmptySelectorSlot
- (void)method:(int)arg;
- (void)method:(int)arg :(int)empty;

- (void)multiple:(int)arg1 args:(int)arg2
                :(int)arg3;
- (void)multiple:(int)arg1 :(int)arg2 args:(int)arg3;
@end

@class WS;
#endif

#if defined(FIRST)
struct S1 {
  Interface1 *I;
  int y;
};
#elif defined(SECOND)
struct S1 {
  Interface1 *I;
  float y;
};
#else
struct S1 s;
// expected-error@second.h:* {{'S1::y' from module 'SecondModule' is not present in definition of 'struct S1' in module 'FirstModule'}}
// expected-note@first.h:* {{declaration of 'y' does not match}}
#endif

#if defined(FIRST)
@interface Interface4 <T : I1 *> {
@public
  T x;
}
@end
@interface Interface5 <T : I1 *> {
@public
  T x;
}
@end
@interface Interface6 <T1 : I1 *, T2 : I2 *> {
@public
  T1 x;
}
@end
#elif defined(SECOND)
@interface Interface4 <T : I1 *> {
@public
  T x;
}
@end
@interface Interface5 <T : I1 *> {
@public
  T x;
}
@end
@interface Interface6 <T1 : I1 *, T2 : I2 *> {
@public
  T2 x;
}
@end
#endif

// Test super class mismatches
#if defined(FIRST)
@interface A1 : I1
@end
#elif defined(SECOND)
@interface A1 : I2
@end
#else
A1 *a1;
// expected-error@first.h:* {{'A1' has different definitions in different modules; first difference is definition in module 'FirstModule' found super class with type 'I1'}}
// expected-note@second.h:* {{but in 'SecondModule' found super class with type 'I2'}}
#endif

#if defined(FIRST)
@interface A2
@end
#elif defined(SECOND)
@interface A2 : I1
@end
#else
// expected-error@first.h:* {{'A2' has different definitions in different modules; first difference is definition in module 'FirstModule' found no super class}}
// expected-note@second.h:* {{but in 'SecondModule' found super class with type 'I1'}}
A2 *a2;
#endif

#if defined(FIRST)
@interface A3 : I1
@end
#elif defined(SECOND)
@interface A3 : I1
@end
#else
A3 *a3;
#endif

#if defined(FIRST)
@interface A4 : I1
@end
#elif defined(SECOND)
@interface A4
@end
#else
A4 *a4;
// expected-error@first.h:* {{'A4' has different definitions in different modules; first difference is definition in module 'FirstModule' found super class with type 'I1'}}
// expected-note@second.h:* {{but in 'SecondModule' found no super class}}
#endif

// Test number of protocols mismatches
#if defined(FIRST)
@interface B1 : NSObject <P1>
@end
#elif defined(SECOND)
@interface B1 : NSObject <P1>
@end
#else
B1 *b1;
#endif

#if defined(FIRST)
@interface B2 : NSObject <P1>
@end
#elif defined(SECOND)
@interface B2 : NSObject
@end
#else
B2 *b2;
// expected-error@first.h:* {{'B2' has different definitions in different modules; first difference is definition in module 'FirstModule' found 1 referenced protocol}}
// expected-note@second.h:* {{but in 'SecondModule' found 0 referenced protocols}}
#endif

#if defined(FIRST)
@interface B3 : NSObject
@end
#elif defined(SECOND)
@interface B3 : NSObject <P1>
@end
#else
B3 *b3;
// expected-error@first.h:* {{'B3' has different definitions in different modules; first difference is definition in module 'FirstModule' found 0 referenced protocols}}
// expected-note@second.h:* {{but in 'SecondModule' found 1 referenced protocol}}
#endif

#if defined(FIRST)
@interface B4 : NSObject <P1>
@end
#elif defined(SECOND)
@interface B4 : NSObject <P2>
@end
#else
// expected-error@first.h:* {{'B4' has different definitions in different modules; first difference is definition in module 'FirstModule' found with 1st protocol named 'P1'}}
// expected-note@second.h:* {{but in 'SecondModule' found with 1st protocol named 'P2'}}
B4 *b4;
#endif

#if defined(FIRST)
@interface M1
- (void)sayHello;
@end
#elif defined(SECOND)
@interface M1
- (int)sayHello;
@end
#else
struct SM1 {
  M1 *x1;
};
struct SM1 sm1;
// expected-error@first.h:* {{'M1' has different definitions in different modules; first difference is definition in module 'FirstModule' found return type is 'void'}}
// expected-note@second.h:* {{but in 'SecondModule' found different return type 'int'}}
#endif

#if defined(FIRST)
@interface M2
- (void)sayHello;
@end
#elif defined(SECOND)
@interface M2
- (void)sayGoodbye;
@end
#else
M2 *m2;
// expected-error@first.h:* {{'M2' has different definitions in different modules; first difference is definition in module 'FirstModule' found method name 'sayHello'}}
// expected-note@second.h:* {{but in 'SecondModule' found method name 'sayGoodbye'}}
#endif

#if defined(FIRST)
@interface M3
- (void)sayHello;
@end
#elif defined(SECOND)
@interface M3
+ (void)sayHello;
@end
#else
M3 *m3;
// expected-error@first.h:* {{'M3' has different definitions in different modules; first difference is definition in module 'FirstModule' found instance method}}
// expected-note@second.h:* {{but in 'SecondModule' found class method}}
#endif

#if defined(FIRST)
@interface SuperM4
- (void)sayHello;
@end
@interface M4 : SuperM4
@end
#elif defined(SECOND)
@interface SuperM4
- (void)sayHello;
@end
@interface M4 : SuperM4
@end
#else
M4 *m4;
#endif

#if defined(FIRST)
@interface MP1
- (void)compute:(int)arg;
@end
#elif defined(SECOND)
@interface MP1
- (void)compute:(float)arg;
@end
#else
MP1 *mp1;
// expected-error@first.h:* {{'MP1' has different definitions in different modules; first difference is definition in module 'FirstModule' found method 'compute:' with 1st parameter of type 'int'}}
// expected-note@second.h:* {{but in 'SecondModule' found method 'compute:' with 1st parameter of type 'float'}}
#endif

#if defined(FIRST)
@interface MP2
- (void)compute:(int)arg0 :(int)arg1;
@end
#elif defined(SECOND)
@interface MP2
- (void)compute:(int)arg0;
@end
#else
MP2 *mp2;
// expected-error@first.h:* {{'MP2' has different definitions in different modules; first difference is definition in module 'FirstModule' found method 'compute::' that has 2 parameters}}
// expected-note@second.h:* {{but in 'SecondModule' found method 'compute:' that has 1 parameter}}
#endif

#if defined(FIRST)
@interface MP3
- (void)compute:(int)arg0;
@end
#elif defined(SECOND)
@interface MP3
- (void)compute:(int)arg1;
@end
#else
MP3 *mp3;
// expected-error@first.h:* {{'MP3' has different definitions in different modules; first difference is definition in module 'FirstModule' found method 'compute:' with 1st parameter named 'arg0'}}
// expected-note@second.h:* {{but in 'SecondModule' found method 'compute:' with 1st parameter named 'arg1'}}
#endif

#if defined(FIRST)
@interface MD
- (id)init __attribute__((objc_designated_initializer));
@end
#elif defined(SECOND)
@interface MD
- (id)init;
@end
#else
MD *md;
// expected-error@first.h:* {{'MD' has different definitions in different modules; first difference is definition in module 'FirstModule' found method with designater initializer}}
// expected-note@second.h:* {{but in 'SecondModule' found method with no designater initializer}}
#endif

#if defined(FIRST)
@interface MDT
- (int)fastBinOp __attribute__((objc_direct));
@end
#elif defined(SECOND)
@interface MDT
- (int)fastBinOp;
@end
#else
MDT *mdt;
// expected-error@first.h:* {{'MDT' has different definitions in different modules; first difference is definition in module 'FirstModule' found direct method}}
// expected-note@second.h:* {{but in 'SecondModule' found no direct method}}
#endif

#if defined(FIRST)
@interface IV1 {
  int a;
}
@end
#elif defined(SECOND)
@interface IV1 {
  char a;
}
@end
#else
IV1 *iv1;
// expected-error@first.h:* {{'IV1' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'a' with type 'int'}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'a' with type 'char'}}
#endif

#if defined(FIRST)
@interface IV2 {
  int a;
}
@end
@interface IV3 {
}
@end
#elif defined(SECOND)
@interface IV2 {
}
@end
@interface IV3 {
  int a;
}
@end
#else
IV2 *iv2;
// expected-error@first.h:* {{'IV2' has different definitions in different modules; first difference is definition in module 'FirstModule' found instance variable}}
// expected-note@second.h:* {{but in 'SecondModule' found end of class}}
IV3 *iv3;
// expected-error@first.h:* {{'IV3' has different definitions in different modules; first difference is definition in module 'FirstModule' found end of class}}
// expected-note@second.h:* {{but in 'SecondModule' found instance variable}}
#endif

#if defined(FIRST)
@interface IV4 {
  int a;
}
@end
@interface IV5 {
@public
  int b;
}
@end
#elif defined(SECOND)
@interface IV4 {
@private
  int a;
}
@end
@interface IV5 {
@package
  int b;
}
@end
#else
IV4 *iv4;
// expected-error@first.h:* {{'IV4' has different definitions in different modules; first difference is definition in module 'FirstModule' found instance variable 'a' access control is @protected}}
// expected-note@second.h:* {{but in 'SecondModule' found instance variable 'a' access control is @private}}
IV5 *iv5;
// expected-error@first.h:* {{'IV5' has different definitions in different modules; first difference is definition in module 'FirstModule' found instance variable 'b' access control is @public}}
// expected-note@second.h:* {{but in 'SecondModule' found instance variable 'b' access control is @package}}
#endif

#if defined(FIRST)
@interface IP1
@property (nonatomic, assign) float f;
@end
#elif defined(SECOND)
@interface IP1
@property (nonatomic, assign) float d;
@end
#else
IP1 *ip1;
// expected-error@first.h:* {{'IP1' has different definitions in different modules; first difference is definition in module 'FirstModule' found property name 'f'}}
// expected-note@second.h:* {{but in 'SecondModule' found property name 'd'}}
#endif

#if defined(FIRST)
@interface IP2
@property (nonatomic, assign) float x;
@end
#elif defined(SECOND)
@interface IP2
@property (nonatomic, assign) int x;
@end
#else
IP2 *ip2;
// expected-error@first.h:* {{'IP2' has different definitions in different modules; first difference is definition in module 'FirstModule' found property 'x' with type 'float'}}
// expected-note@second.h:* {{but in 'SecondModule' found property 'x' with type 'int'}}
#endif

#if defined(FIRST)
@class NSArray;
@interface IP3
@property (assign, readwrite, atomic, unsafe_unretained) NSArray* NA;
@end
@interface IP4
@property () NSArray* NA; // empty list defaults to: readwrite atomic strong
@end
@interface IP5
@property () NSArray* NA;
@end
@interface IP6
@property () NSArray* NA;
@end
@interface IP7
@property (weak) NSArray* NA;
@end
@interface IP8
@property (strong) NSArray* NA;
@end
#elif defined(SECOND)
@class NSArray;
@interface IP3
@property () NSArray* NA;
@end
@interface IP4
@property (assign, readwrite, atomic, unsafe_unretained) NSArray* NA;
@end
@interface IP5
@property () NSArray* NA;
@end
@interface IP6
@property (readwrite, atomic, strong) NSArray* NA;
@end
@interface IP7
@property (strong) NSArray* NA;
@end
@interface IP8
@property (weak) NSArray* NA;
@end
#else
IP3 *ip3;
// expected-error@first.h:* {{'IP3' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'assign' property attribute}}
// expected-note@second.h:* {{but in 'SecondModule' found no written or default attribute for property}}
IP4 *ip4;
// expected-error@second.h:* {{'IP4' has different definitions in different modules; first difference is definition in module 'SecondModule' found 'assign' property attribute}}
// expected-note@first.h:* {{but in 'FirstModule' found no written or default attribute for property}}
IP5 *ip5;
IP6 *ip6;
IP7 *ip7;
// expected-error@first.h:* {{'IP7' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'weak' property attribute}}
// expected-note@second.h:* {{but in 'SecondModule' found no written or default attribute for property}}
IP8 *ip8;
// expected-error@second.h:* {{'IP8' has different definitions in different modules; first difference is definition in module 'SecondModule' found 'weak' property attribute}}
// expected-note@first.h:* {{but in 'FirstModule' found no written or default attribute for property}}
#endif

#if defined(FIRST)
@protocol W1 <P1>
@end
@interface WI1 <P1>
@end
#elif defined(SECOND)
@protocol W1 <P2>
@end
@interface WI1 <P2>
@end
#else
@interface IFP1 <W1>
@end
IFP1 *ifp1;
// expected-error@first.h:* {{'W1' has different definitions in different modules; first difference is definition in module 'FirstModule' found with 1st protocol named 'P1'}}
// expected-note@second.h:* {{but in 'SecondModule' found with 1st protocol named 'P2'}}
WI1 *wi1;
// expected-error@first.h:* {{'WI1' has different definitions in different modules; first difference is definition in module 'FirstModule' found with 1st protocol named 'P1'}}
// expected-note@second.h:* {{but in 'SecondModule' found with 1st protocol named 'P2'}}
#endif

#if defined(FIRST)
@protocol W2
@property (nonatomic, assign) float f;
@end
@protocol W3
@optional
@property (nonatomic, assign) int a;
@end
@protocol W4
@required
@property (nonatomic, assign) int a;
@end
@protocol W5
@required
@property (nonatomic, assign) int a;
@end
#elif defined(SECOND)
@protocol W2
@property (nonatomic, assign) float d;
@end
@protocol W3
@property (nonatomic, assign) int a;
@end
@protocol W4
@optional
@property (nonatomic, assign) int a;
@end
@protocol W5
@property (nonatomic, assign) int a;
@end
#else
@interface IW2 <W2>
@end
// expected-error@first.h:* {{'W2' has different definitions in different modules; first difference is definition in module 'FirstModule' found property name 'f'}}
// expected-note@second.h:* {{but in 'SecondModule' found property name 'd'}}
@interface IW3 <W3>
@end
// expected-error@first.h:* {{'W3' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'optional' property control}}
// expected-note@second.h:* {{but in 'SecondModule' found no property control}}
@interface IW4 <W4>
@end
// expected-error@first.h:* {{'W4' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'required' property control}}
// expected-note@second.h:* {{but in 'SecondModule' found 'optional' property control}}
@interface IW5 <W5>
@end
// expected-error@first.h:* {{'W5' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'required' property control}}
// expected-note@second.h:* {{but in 'SecondModule' found no property control}}
#endif

#if defined(FIRST)
@protocol MW2
- (void)compute:(int)arg0;
@end
@protocol MW3
@optional
- (void)compute:(int)arg0;
@end
@protocol MW4
@required
- (void)compute:(int)arg0;
@end
@protocol MW5
@required
- (void)compute:(int)arg0;
@end
#elif defined(SECOND)
@protocol MW2
- (void)compute:(int)arg1;
@end
@protocol MW3
- (void)compute:(int)arg0;
@end
@protocol MW4
@optional
- (void)compute:(int)arg0;
@end
@protocol MW5
- (void)compute:(int)arg0;
@end
#else
@interface IMW2 <MW2>
@end
// expected-error@first.h:* {{'MW2' has different definitions in different modules; first difference is definition in module 'FirstModule' found method 'compute:' with 1st parameter named 'arg0'}}
// expected-note@second.h:* {{but in 'SecondModule' found method 'compute:' with 1st parameter named 'arg1'}}
@interface IMW3 <MW3>
@end
// expected-error@first.h:* {{'MW3' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'optional' method control}}
// expected-note@second.h:* {{but in 'SecondModule' found 'required' method control}}
@interface IMW4 <MW4>
@end
// expected-error@first.h:* {{'MW4' has different definitions in different modules; first difference is definition in module 'FirstModule' found 'required' method control}}
// expected-note@second.h:* {{but in 'SecondModule' found 'optional' method control}}
@interface IMW5 <MW5> // No diagnostics: @required is the default.
@end
#endif

#if defined(FIRST)
@protocol PP1 <UP1>
@end
#elif defined(SECOND)
@protocol UP1
@end
@protocol PP11 <UP1>
@end
#else
@interface II0 <PP1>
@end
II0 *ii0;
#endif

#if defined(FIRST)
@protocol PP2 <UP2>
@end
#elif defined(SECOND)
@protocol PP2 <UP2>
@end
#else
#endif

#if defined(FIRST)
@protocol PP3 <UP2>
@end
#elif defined(SECOND)
@protocol PP3 <UP3>
@end
#else
@protocol PP4 <PP3>
@end
// expected-error@first.h:* {{'PP3' has different definitions in different modules; first difference is definition in module 'FirstModule' found with 1st protocol named 'UP2'}}
// expected-note@second.h:* {{but in 'SecondModule' found with 1st protocol named 'UP3'}}
#endif

#if defined(FIRST)
@interface II1 <UP3>
@end
// expected-warning@first.h:* {{cannot find protocol definition for 'UP3'}}
// expected-note@first.h:* {{protocol 'UP3' has no definition}}
#elif defined(SECOND)
@protocol UP3
@end
@interface II1 <UP3>
@end
#else
#endif

#if defined(FIRST)
@interface IP9
@property (nonatomic, readonly, strong) WS * _Nullable ws;
@end
#elif defined(SECOND)
@interface WS
- (void)sayHello;
@end
@interface IP9
@property (nonatomic, readonly, strong) WS * ws;
@end
#else
IP9 *ip9;
#endif


// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
