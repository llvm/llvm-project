// XFAIL: *
// TODO: Remove or cut it down to one symbol rename.

@interface I1

@property int p1; // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
@property (readwrite, nonatomic) int p2; // CHECK2: rename [[@LINE]]:38 -> [[@LINE]]:40

@end

// RUN: clang-refactor-test rename-initiate -at=%s:3:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:4:38 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

@implementation I1

- (void)foo:(I1 *)other {
  self.p2 =          // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10
            self.p1; // CHECK1: rename [[@LINE]]:18 -> [[@LINE]]:20
  (void)other.p1;    // CHECK1: rename [[@LINE]]:15 -> [[@LINE]]:17
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:14:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:15:18 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:16:15 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

@implementation I1 (gettersAndSetters)

- (void)foo2:(I1 *)other {
  int x = [self p1]; // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
  [self setP1: x];   // CHECK1: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14
  [other setP2:      // CHECK2: rename "setFoo" [[@LINE]]:10 -> [[@LINE]]:15
    [other p2]];     // CHECK2: rename [[@LINE]]:12 -> [[@LINE]]:14
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:28:17 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:29:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:30:10 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:31:12 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

@interface I2

@property int noImplementation; // CHECK3: rename [[@LINE]]:15 -> [[@LINE]]:31

@end

void noImplementationGetterSetters(I2 *object) {
  object.noImplementation = 2;     // CHECK3: rename [[@LINE]]:10 -> [[@LINE]]:26
  int x = object.noImplementation; // CHECK3: rename [[@LINE]]:18 -> [[@LINE]]:34
  [object setNoImplementation: x]; // CHECK3: rename "setFoo" [[@LINE]]:11 -> [[@LINE]]:30
  (void)[object noImplementation]; // CHECK3: rename [[@LINE]]:17 -> [[@LINE]]:33
}

// RUN: clang-refactor-test rename-initiate -at=%s:43:15 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:48:10 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:49:18 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:50:11 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:51:17 -new-name=foo %s | FileCheck --check-prefix=CHECK3 %s

@interface I3

@property (readonly) int noSetter; // CHECK4: rename [[@LINE]]:26 -> [[@LINE]]:34

@end

void noPropertySetter(I3 *object) {
  (void)object.noSetter;   // CHECK4: rename [[@LINE]]:16 -> [[@LINE]]:24
  object.noSetter = 2;     // CHECK4-NOT: rename [[@LINE]]
  (void)[object noSetter]; // CHECK4: rename [[@LINE]]:17 -> [[@LINE]]:25
  [object setNoSetter: 2]; // CHECK4-NOT: rename "setFoo" [[@LINE]]
}

// RUN: clang-refactor-test rename-initiate -at=%s:62:26 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:67:16 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:69:17 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s

@interface PropertyOverrides1: I1

- (int)p1; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
- (void)setP1:(int)x; // CHECK1: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14
- (int)p2; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10

@end

@implementation PropertyOverrides1 {
  I1 *object;
}

- (int)p1 {            // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:10
  return [super p1];   // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:19
}
- (void)setP1:(int)x { // CHECK1: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14
  [object setP1: x];   // CHECK1: rename "setFoo" [[@LINE]]:11 -> [[@LINE]]:16
  [super setP1: x];    // CHECK1: rename "setFoo" [[@LINE]]:10 -> [[@LINE]]:15
}
- (int)p2 {            // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:10
  return super.p2;     // CHECK2: rename [[@LINE]]:16 -> [[@LINE]]:18
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:79:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:80:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:89:8 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:90:17 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:92:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:93:11 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:94:10 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:81:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:96:8 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:97:16 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

@interface PropertyOverrides2: I3

- (int)noSetter; // CHECK4: rename [[@LINE]]:8 -> [[@LINE]]:16
- (void)setNoSetter:(int)x; // CHECK4-NOT: rename "setFoo" [[@LINE]]

@end

void getterOnlyOverrideWithoutImplementation(PropertyOverrides2 *object) {
  (void)object.noSetter;     // CHECK4: rename [[@LINE]]:16 -> [[@LINE]]:24
  int _ = [object noSetter]; // CHECK4: rename [[@LINE]]:19 -> [[@LINE]]:27
}

// RUN: clang-refactor-test rename-initiate -at=%s:115:8 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:121:16 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:122:19 -new-name=foo %s | FileCheck --check-prefix=CHECK4 %s

@interface MismatchedPropertyOverrides: I1

- (void)p1; // CHECK1: rename [[@LINE]]:9 -> [[@LINE]]:11

@end

// RUN: clang-refactor-test rename-initiate -at=%s:131:9 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s

@interface ExplicitlyNamedGetterSetters1

@property (getter=getP3) int p3; // CHECK5: rename [[@LINE]]:30 -> [[@LINE]]:32
                                 // CHECK5GET: rename [[@LINE-1]]:19 -> [[@LINE-1]]:24
@property (getter=a, setter=b:) int p4; // CHECK6: rename [[@LINE]]:37 -> [[@LINE]]:39
                                        // CHECK6GET: rename [[@LINE-1]]:19 -> [[@LINE-1]]:20
                                        // CHECK6SET: rename [[@LINE-2]]:29 -> [[@LINE-2]]:30
@property (readonly, getter=local) bool isLocal; // CHECK7: rename [[@LINE]]:41 -> [[@LINE]]:48
                                                 // CHECK7GET: rename [[@LINE-1]]:29 -> [[@LINE-1]]:34
@end

@implementation ExplicitlyNamedGetterSetters1

- (void)foo:(ExplicitlyNamedGetterSetters *)other {
  self.p3 = 2; // CHECK5: rename [[@LINE]]:8 -> [[@LINE]]:10
  [self p3];   // CHECK5-NOT: rename [[@LINE]]
  [self setP3: // CHECK5: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14
    [other getP3]]; // CHECK5-NOT: rename [[@LINE]]
                    // CHECK5GET: rename [[@LINE-1]]:12 -> [[@LINE-1]]:17

  self.p4 = 3; // CHECK6: rename [[@LINE]]:8 -> [[@LINE]]:10
  [self p4];   // CHECK6-NOT: rename [[@LINE]]
  [self setP4: 2]; // CHECK6-NOT: rename "setFoo" [[@LINE]]
  [self b:         // CHECK6-NOT: rename "setFoo" [[@LINE]]
                   // CHECK6SET: rename [[@LINE-1]]:9 -> [[@LINE-1]]:10
    [other a]];    // CHECK6-NOT: rename [[@LINE]]
                   // CHECK6GET: rename [[@LINE-1]]:12 -> [[@LINE-1]]:13

  (void)self.isLocal; // CHECK7: rename [[@LINE]]:14 -> [[@LINE]]:21
  [self isLocal]; // CHECK7-NOT: rename [[@LINE]]
  [other local];  // CHECK7-NOT: rename [[@LINE]]
                  // CHECK7GET: rename [[@LINE-1]]:10 -> [[@LINE-1]]:15
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:139:30 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:151:8 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:153:9 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s

// RUN: clang-refactor-test rename-initiate -at=%s:141:37 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-initiate -at=%s:157:8 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s

// RUN: clang-refactor-test rename-initiate -at=%s:144:41 -new-name=foo %s | FileCheck --check-prefix=CHECK7 %s
// RUN: clang-refactor-test rename-initiate -at=%s:165:14 -new-name=foo %s | FileCheck --check-prefix=CHECK7 %s

// RUN: clang-refactor-test rename-initiate -at=%s:139:19 -new-name=foo %s | FileCheck --check-prefix=CHECK5GET %s
// RUN: clang-refactor-test rename-initiate -at=%s:154:12 -new-name=foo %s | FileCheck --check-prefix=CHECK5GET %s

// RUN: clang-refactor-test rename-initiate -at=%s:141:19 -new-name=foo %s | FileCheck --check-prefix=CHECK6GET %s
// RUN: clang-refactor-test rename-initiate -at=%s:162:12 -new-name=foo %s | FileCheck --check-prefix=CHECK6GET %s
// RUN: clang-refactor-test rename-initiate -at=%s:141:29 -new-name=foo %s | FileCheck --check-prefix=CHECK6SET %s
// RUN: clang-refactor-test rename-initiate -at=%s:160:9 -new-name=foo %s | FileCheck --check-prefix=CHECK6SET %s

// RUN: clang-refactor-test rename-initiate -at=%s:144:29 -new-name=foo %s | FileCheck --check-prefix=CHECK7GET %s
// RUN: clang-refactor-test rename-initiate -at=%s:167:10 -new-name=foo %s | FileCheck --check-prefix=CHECK7GET %s

void ivars1(I1 *object) {
  object->_p1 = 2; // CHECK1: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:14
  object->_p2 =    // CHECK2: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:14
                object->_p1;  // CHECK1: rename "_foo" [[@LINE]]:25 -> [[@LINE]]:28
}

void ivars2(ExplicitlyNamedGetterSetters1 *object) {
  object->_p3 = // CHECK5: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:14
                object->_p4; // CHECK6: rename "_foo" [[@LINE]]:25 -> [[@LINE]]:28
  object->_isLocal = 0; // CHECK7: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:19
}

void ivarsNoImplementation(I2 *object) {
  object->_noImplementation = 4; // CHECK2-NOT: rename "_foo" [[@LINE]]
}

// RUN: clang-refactor-test rename-initiate -at=%s:195:11 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:197:25 -new-name=foo %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:196:11 -new-name=foo %s | FileCheck --check-prefix=CHECK2 %s

// RUN: clang-refactor-test rename-initiate -at=%s:201:11 -new-name=foo %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:202:25 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-initiate -at=%s:203:11 -new-name=foo %s | FileCheck --check-prefix=CHECK7 %s

@interface ExplicitIVars

@property int p5; // CHECK8: rename [[@LINE]]:15 -> [[@LINE]]:17
@property(readonly) int p6; // CHECK9: rename [[@LINE]]:25 -> [[@LINE]]:27

@end

@implementation ExplicitIVars {
  int _p5; // CHECK8: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
  int _p6; // CHECK9: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
}

- (void)foo:(ExplicitIVars *)other {
  _p5 =              // CHECK8: rename "_foo" [[@LINE]]:3 -> [[@LINE]]:6
        other->_p6;  // CHECK9: rename "_foo" [[@LINE]]:16 -> [[@LINE]]:19
  other->_p6 =       // CHECK9: rename "_foo" [[@LINE]]:10 -> [[@LINE]]:13
               _p5;  // CHECK8: rename "_foo" [[@LINE]]:16 -> [[@LINE]]:19
  self.p5 =          // CHECK8: rename [[@LINE]]:8 -> [[@LINE]]:10
           other.p6; // CHECK9: rename [[@LINE]]:18 -> [[@LINE]]:20
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:220:15 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-initiate -at=%s:226:7 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-initiate -at=%s:231:3 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-initiate -at=%s:234:16 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-initiate -at=%s:235:8 -new-name=foo %s | FileCheck --check-prefix=CHECK8 %s

// RUN: clang-refactor-test rename-initiate -at=%s:221:25 -new-name=foo %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-initiate -at=%s:227:7 -new-name=foo %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-initiate -at=%s:232:16 -new-name=foo %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-initiate -at=%s:233:10 -new-name=foo %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-initiate -at=%s:236:18 -new-name=foo %s | FileCheck --check-prefix=CHECK9 %s

@interface ExplicitIVarsInInterface {
  int _p7; // CHECK10: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
  @public
  int _p8; // CHECK11: rename "_foo" [[@LINE]]:7 -> [[@LINE]]:10
}

@property int p7; // CHECK10: rename [[@LINE]]:15 -> [[@LINE]]:17
@property int p8; // CHECK11: rename [[@LINE]]:15 -> [[@LINE]]:17

@end

@implementation ExplicitIVarsInInterface
@end

void explicitIVarsInInterface(ExplicitIVarsInInterface* object) {
  object->_p7 = // CHECK10: rename "_foo" [[@LINE]]:11 -> [[@LINE]]:14
                object->_p8; // CHECK11: rename "_foo" [[@LINE]]:25 -> [[@LINE]]:28
}

// RUN: clang-refactor-test rename-initiate -at=%s:254:7 -new-name=foo %s | FileCheck --check-prefix=CHECK10 %s
// RUN: clang-refactor-test rename-initiate -at=%s:259:15 -new-name=foo %s | FileCheck --check-prefix=CHECK10 %s
// RUN: clang-refactor-test rename-initiate -at=%s:268:11 -new-name=foo %s | FileCheck --check-prefix=CHECK10 %s

// RUN: clang-refactor-test rename-initiate -at=%s:256:7 -new-name=foo %s | FileCheck --check-prefix=CHECK11 %s
// RUN: clang-refactor-test rename-initiate -at=%s:260:15 -new-name=foo %s | FileCheck --check-prefix=CHECK11 %s
// RUN: clang-refactor-test rename-initiate -at=%s:269:25 -new-name=foo %s | FileCheck --check-prefix=CHECK11 %s

@interface GetterSetterDefinedInInterfaceOnly

@property int p9; // CHECK12: rename [[@LINE]]:15 -> [[@LINE]]:17

@end

@implementation GetterSetterDefinedInInterfaceOnly

- (int)p9 { return 0; }  // CHECK12: rename [[@LINE]]:8 -> [[@LINE]]:10
- (void)setP9:(int)x { } // CHECK12: rename "setFoo" [[@LINE]]:9 -> [[@LINE]]:14

@end

// RUN: clang-refactor-test rename-initiate -at=%s:288:8 -new-name=foo %s | FileCheck --check-prefix=CHECK12 %s
// RUN: clang-refactor-test rename-initiate -at=%s:289:9 -new-name=foo %s | FileCheck --check-prefix=CHECK12 %s

void matchingGetterSetterSelector() {
  @selector(p1);     // CHECK1: selector [[@LINE]]:13 -> [[@LINE]]:15
  @selector(setP1:); // CHECK1: selector "setFoo" [[@LINE]]:13 -> [[@LINE]]:18
  @selector(setP1);  // CHECK1-NOT: selector "setFoo" [[@LINE]]
}
