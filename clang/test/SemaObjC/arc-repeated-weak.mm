// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -fblocks -Wno-objc-root-class -std=c++11 -Warc-repeated-use-of-weak -verify %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-weak -fblocks -Wno-objc-root-class -std=c++11 -Warc-repeated-use-of-weak -verify %s

@interface Test {
@public
  Test *ivar;
  __weak id weakIvar;
}
@property(weak) Test *weakProp;
@property(strong) Test *strongProp;

- (__weak id)implicitProp;

+ (__weak id)weakProp;
@end

extern void use(id);
extern id get();
extern bool condition();
#define nil ((id)0)

void sanity(Test *a) {
  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this function but may be unpredictably set to nil; assign to a strong variable to keep the object alive}}
  use(a.weakProp); // expected-note{{also accessed here}}

  use(a.strongProp);
  use(a.strongProp); // no-warning

  use(a.weakProp); // expected-note{{also accessed here}}
}

void singleUse(Test *a) {
  use(a.weakProp); // no-warning
  use(a.strongProp); // no-warning
}

void assignsOnly(Test *a) {
  a.weakProp = get(); // no-warning

  id next = get();
  if (next)
    a.weakProp = next; // no-warning

  a->weakIvar = get(); // no-warning
  next = get();
  if (next)
    a->weakIvar = next; // no-warning

  extern __weak id x;
  x = get(); // no-warning
  next = get();
  if (next)
    x = next; // no-warning
}

void assignThenRead(Test *a) {
  a.weakProp = get(); // expected-note{{also accessed here}}
  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
}

void twoVariables(Test *a, Test *b) {
  use(a.weakProp); // no-warning
  use(b.weakProp); // no-warning
}

void doubleLevelAccess(Test *a) {
  use(a.strongProp.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times in this function and may be unpredictably set to nil; assign to a strong variable to keep the object alive}}
  use(a.strongProp.weakProp); // expected-note{{also accessed here}}
}

void doubleLevelAccessIvar(Test *a) {
  use(a.strongProp.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
  use(a.strongProp.weakProp); // expected-note{{also accessed here}}
}

void implicitProperties(Test *a) {
  use(a.implicitProp); // expected-warning{{weak implicit property 'implicitProp' is accessed multiple times}}
  use(a.implicitProp); // expected-note{{also accessed here}}
}

void classProperties() {
  use(Test.weakProp); // expected-warning{{weak implicit property 'weakProp' is accessed multiple times}}
  use(Test.weakProp); // expected-note{{also accessed here}}
}

void classPropertiesAreDifferent(Test *a) {
  use(Test.weakProp); // no-warning
  use(a.weakProp); // no-warning
  use(a.strongProp.weakProp); // no-warning
}

void ivars(Test *a) {
  use(a->weakIvar); // expected-warning{{weak instance variable 'weakIvar' is accessed multiple times}}
  use(a->weakIvar); // expected-note{{also accessed here}}
}

void globals() {
  extern __weak id a;
  use(a); // expected-warning{{weak variable 'a' is accessed multiple times}}
  use(a); // expected-note{{also accessed here}}
}

void messageGetter(Test *a) {
  use([a weakProp]); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  use([a weakProp]); // expected-note{{also accessed here}}
}

void messageSetter(Test *a) {
  [a setWeakProp:get()]; // no-warning
  [a setWeakProp:get()]; // no-warning
}

void messageSetterAndGetter(Test *a) {
  [a setWeakProp:get()]; // expected-note{{also accessed here}}
  use([a weakProp]); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
}

void mixDotAndMessageSend(Test *a, Test *b) {
  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  use([a weakProp]); // expected-note{{also accessed here}}

  use([b weakProp]); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  use(b.weakProp); // expected-note{{also accessed here}}
}


void assignToStrongWrongInit(Test *a) {
  id val = a.weakProp; // expected-note{{also accessed here}}
  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
}

void assignToStrongWrong(Test *a) {
  id val;
  val = a.weakProp; // expected-note{{also accessed here}}
  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
}

void assignToIvarWrong(Test *a) {
  a->weakIvar = get(); // expected-note{{also accessed here}}
  use(a->weakIvar); // expected-warning{{weak instance variable 'weakIvar' is accessed multiple times}}
}

void assignToGlobalWrong() {
  extern __weak id a;
  a = get(); // expected-note{{also accessed here}}
  use(a); // expected-warning{{weak variable 'a' is accessed multiple times}}
}

void assignToStrongOK(Test *a) {
  if (condition()) {
    id val = a.weakProp; // no-warning
    (void)val;
  } else {
    id val;
    val = a.weakProp; // no-warning
    (void)val;
  }
}

void assignToStrongConditional(Test *a) {
  id val = (condition() ? a.weakProp : a.weakProp); // no-warning
  id val2 = a.implicitProp ?: a.implicitProp; // no-warning
}

void testBlock(Test *a) {
  use(a.weakProp); // no-warning

  use(^{
    use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this block}}
    use(a.weakProp); // expected-note{{also accessed here}}
  });
}

void assignToStrongWithCasts(Test *a) {
  if (condition()) {
    Test *val = (Test *)a.weakProp; // no-warning
    (void)val;
  } else {
    id val;
    val = (Test *)a.weakProp; // no-warning
    (void)val;
  }
}

void assignToStrongWithMessages(Test *a) {
  if (condition()) {
    id val = [a weakProp]; // no-warning
    (void)val;
  } else {
    id val;
    val = [a weakProp]; // no-warning
    (void)val;
  }
}


void assignAfterRead(Test *a) {
  // Special exception for a single read before any writes.
  if (!a.weakProp) // no-warning
    a.weakProp = get(); // no-warning
}

void readOnceWriteMany(Test *a) {
  if (!a.weakProp) { // no-warning
    a.weakProp = get(); // no-warning
    a.weakProp = get(); // no-warning
  }
}

void readOnceAfterWrite(Test *a) {
  a.weakProp = get(); // expected-note{{also accessed here}}
  if (!a.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
    a.weakProp = get(); // expected-note{{also accessed here}}
  }
}

void readOnceWriteManyLoops(Test *a, Test *b, Test *c, Test *d, Test *e) {
  while (condition()) {
    if (!a.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
      a.weakProp = get(); // expected-note{{also accessed here}}
      a.weakProp = get(); // expected-note{{also accessed here}}
    }
  }

  do {
    if (!b.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
      b.weakProp = get(); // expected-note{{also accessed here}}
      b.weakProp = get(); // expected-note{{also accessed here}}
    }
  } while (condition());

  for (id x = get(); x; x = get()) {
    if (!c.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
      c.weakProp = get(); // expected-note{{also accessed here}}
      c.weakProp = get(); // expected-note{{also accessed here}}
    }
  }

  for (id x in get()) {
    if (!d.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
      d.weakProp = get(); // expected-note{{also accessed here}}
      d.weakProp = get(); // expected-note{{also accessed here}}
    }
  }

  int array[] = { 1, 2, 3 };
  for (int i : array) {
    if (!e.weakProp) { // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
      e.weakProp = get(); // expected-note{{also accessed here}}
      e.weakProp = get(); // expected-note{{also accessed here}}
    }
  }
}

void readOnlyLoop(Test *a) {
  while (condition()) {
    use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
  }
}

void readInIterationLoop() {
  for (Test *a in get())
    use(a.weakProp); // no-warning
}

void readDoubleLevelAccessInLoop() {
  for (Test *a in get()) {
    use(a.strongProp.weakProp); // no-warning
  }
}

void readParameterInLoop(Test *a) {
  for (id unused in get()) {
    use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this function}}
    (void)unused;
  }
}

void readGlobalInLoop() {
  static __weak id a;
  for (id unused in get()) {
    use(a); // expected-warning{{weak variable 'a' is accessed multiple times in this function}}
    (void)unused;
  }
}

void doWhileLoop(Test *a) {
  do {
    use(a.weakProp); // no-warning
  } while(0);
}


@interface Test (Methods)
@end

@implementation Test (Methods)
- (void)sanity {
  use(self.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this method but may be unpredictably set to nil; assign to a strong variable to keep the object alive}}
  use(self.weakProp); // expected-note{{also accessed here}}
}

- (void)ivars {
  use(weakIvar); // expected-warning{{weak instance variable 'weakIvar' is accessed multiple times in this method but may be unpredictably set to nil; assign to a strong variable to keep the object alive}}
  use(weakIvar); // expected-note{{also accessed here}}
}

- (void)doubleLevelAccessForSelf {
  use(self.strongProp.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  use(self.strongProp.weakProp); // expected-note{{also accessed here}}

  use(self->ivar.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  use(self->ivar.weakProp); // expected-note{{also accessed here}}

  use(self->ivar->weakIvar); // expected-warning{{weak instance variable 'weakIvar' is accessed multiple times}}
  use(self->ivar->weakIvar); // expected-note{{also accessed here}}
}

- (void)distinctFromOther:(Test *)other {
  use(self.strongProp.weakProp); // no-warning
  use(other.strongProp.weakProp); // no-warning

  use(self->ivar.weakProp); // no-warning
  use(other->ivar.weakProp); // no-warning

  use(self.strongProp->weakIvar); // no-warning
  use(other.strongProp->weakIvar); // no-warning
}
@end

@interface Base1
@end
@interface Sub1 : Base1
@end
@interface Sub1(cat)
-(id)prop;
@end

void test1(Sub1 *s) {
  use([s prop]);
  use([s prop]);
}

@interface Base1(cat)
@property (weak) id prop;
@end

void test2(Sub1 *s) {
  // This does not warn because the "prop" in "Base1(cat)" was introduced
  // after the method declaration and we don't find it as overridden.
  // Always looking for overridden methods after the method declaration is expensive
  // and it's not clear it is worth it currently.
  use([s prop]);
  use([s prop]);
}


class Wrapper {
  Test *a;

public:
  void fields() {
    use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times in this function but may be unpredictably set to nil; assign to a strong variable to keep the object alive}}
    use(a.weakProp); // expected-note{{also accessed here}}
  }

  void distinctFromOther(Test *b, const Wrapper &w) {
    use(a.weakProp); // no-warning
    use(b.weakProp); // no-warning
    use(w.a.weakProp); // no-warning
  }

  static void doubleLevelAccessField(const Wrapper &x, const Wrapper &y) {
    use(x.a.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
    use(y.a.weakProp); // expected-note{{also accessed here}}
  }
};


// -----------------------
// False positives
// -----------------------

// Most of these would require flow-sensitive analysis to silence correctly.

void assignNil(Test *a) {
  if (condition())
    a.weakProp = nil; // expected-note{{also accessed here}}

  use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
}

void branch(Test *a) {
  if (condition())
    use(a.weakProp); // expected-warning{{weak property 'weakProp' is accessed multiple times}}
  else
    use(a.weakProp); // expected-note{{also accessed here}}
}

void doubleLevelAccess(Test *a, Test *b) {
  use(a.strongProp.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
  use(b.strongProp.weakProp); // expected-note{{also accessed here}}

  use(a.weakProp.weakProp); // no-warning
}

void doubleLevelAccessIvar(Test *a, Test *b) {
  use(a->ivar.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
  use(b->ivar.weakProp); // expected-note{{also accessed here}}

  use(a.strongProp.weakProp); // no-warning
}

// rdar://13942025
@interface X
@end

@implementation X
- (int) warningAboutWeakVariableInsideTypeof {
    __typeof__(self) __weak weakSelf = self;
    ^(){
        __typeof__(weakSelf) blockSelf = weakSelf;
        use(blockSelf);
    }();
    return sizeof(weakSelf);
}
@end

// rdar://19053620
@interface NSNull
+ (NSNull *)null;
@end

@interface INTF @end

@implementation INTF
- (void) Meth : (id) data
{
  data = data ?: NSNull.null;
}
@end

// This used to crash in WeakObjectProfileTy::getBaseInfo when getBase() was
// called on an ObjCPropertyRefExpr object whose receiver was an interface.

@class NSString;
@interface NSBundle
+(NSBundle *)foo;
@property (class, strong) NSBundle *foo2;
@property (strong) NSString *prop;
@property(weak) NSString *weakProp;
@end

@interface NSBundle2 : NSBundle
@end

void foo() {
  NSString * t = NSBundle.foo.prop;
  use(NSBundle.foo.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
  use(NSBundle2.foo.weakProp); // expected-note{{also accessed here}}

  NSString * t2 = NSBundle.foo2.prop;
  use(NSBundle.foo2.weakProp); // expected-warning{{weak property 'weakProp' may be accessed multiple times}}
  use(NSBundle2.foo2.weakProp); // expected-note{{also accessed here}}
  decltype([NSBundle2.foo2 weakProp]) t3;
  decltype(NSBundle2.foo2.weakProp) t4;
  __typeof__(NSBundle2.foo2.weakProp) t5;
}

// This used to crash in the constructor of WeakObjectProfileTy when a
// DeclRefExpr was passed that didn't reference a VarDecl.

typedef INTF * INTFPtrTy;

enum E {
  e1
};

void foo1() {
  INTFPtrTy tmp = (INTFPtrTy)e1;
#if __has_feature(objc_arc)
// expected-error@-2{{cast of 'E' to 'INTFPtrTy' (aka 'INTF *') is disallowed with ARC}}
#endif
}

@class NSString;
static NSString* const kGlobal = @"";

@interface NSDictionary
- (id)objectForKeyedSubscript:(id)key;
@end

@interface WeakProp
@property (weak) NSDictionary *nd;
@end

@implementation WeakProp
-(void)m {
  (void)self.nd[@""]; // no warning
}
@end
