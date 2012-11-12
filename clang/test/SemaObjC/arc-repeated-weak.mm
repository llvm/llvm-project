// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -fblocks -Wno-objc-root-class -std=c++11 -Warc-repeated-use-of-weak -verify %s

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

