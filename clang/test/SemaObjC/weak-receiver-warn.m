// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -fblocks -Wno-objc-root-class -Wreceiver-is-weak -verify %s
// rdar://10225276

@interface Test0
- (void) setBlock: (void(^)(void)) block;
- (void) addBlock: (void(^)(void)) block;
- (void) actNow;
@end

void test0(Test0 *x) {
  __weak Test0 *weakx = x;
  [x addBlock: ^{ [weakx actNow]; }]; // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
  [x setBlock: ^{ [weakx actNow]; }]; // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
  x.block = ^{ [weakx actNow]; }; // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}

  [weakx addBlock: ^{ [x actNow]; }]; // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
  [weakx setBlock: ^{ [x actNow]; }]; // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
  weakx.block = ^{ [x actNow]; };     // expected-warning {{weak receiver may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
}

@interface Test
{
  __weak Test* weak_prop;
}
- (void) Meth;
@property  __weak Test* weak_prop; // expected-note {{property declared here}}
@property (weak, atomic) id weak_atomic_prop; // expected-note {{property declared here}}
- (__weak id) P; // expected-note {{method 'P' declared here}}
@end

@implementation Test
- (void) Meth {
    if (self.weak_prop) {
      self.weak_prop = 0;
    }
    if (self.weak_atomic_prop) {
      self.weak_atomic_prop = 0;
    }
    [self.weak_prop Meth]; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
    id pi = self.P;

    [self.weak_atomic_prop Meth];  // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}

    [self.P Meth];		   // expected-warning {{weak implicit property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
}

- (__weak id) P { return 0; }
@dynamic weak_prop, weak_atomic_prop;
@end


@interface MyClass {
    __weak MyClass *_parent;
}
@property (weak) MyClass *parent; // expected-note 4 {{property declared here}}
@end

@implementation MyClass
@synthesize parent = _parent;

- (void)doSomething
{
    [[self parent] doSomething]; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}

    (void)self.parent.doSomething; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
}

@end


// Weak properties on protocols can be synthesized by an adopting class.
@protocol MyProtocol
@property (weak) id object; // expected-note 2 {{property declared here}}
@end

void testProtocol(id <MyProtocol> input) {
  [[input object] Meth]; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
  [input.object Meth]; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
}


@interface Subclass : MyClass
// Unnecessarily redeclare -parent.
- (id)parent;
@end

@implementation Subclass

- (id)parent {
  return [super parent];
}

- (void)doSomethingElse {
  [[self parent] doSomething]; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}

  (void)self.parent.doSomething; // expected-warning {{weak property may be unpredictably set to nil}} expected-note {{assign the value to a strong variable to keep the object alive during use}}
}

@end

