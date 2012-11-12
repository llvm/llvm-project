// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -fblocks -verify -Wno-objc-root-class -Wno-implicit-retain-self %s

void *_Block_copy(const void *block);

@interface Test0
- (void) setBlock: (void(^)(void)) block;
- (void) addBlock: (void(^)(void)) block;
- (void) actNow;
@end
void test0(Test0 *x) {
  [x setBlock: // expected-note {{block will be retained by the captured object}}
       ^{ [x actNow]; }]; // expected-warning {{capturing 'x' strongly in this block is likely to lead to a retain cycle}}
  x.block = // expected-note {{block will be retained by the captured object}}
       ^{ [x actNow]; }; // expected-warning {{capturing 'x' strongly in this block is likely to lead to a retain cycle}}

  [x addBlock: // expected-note {{block will be retained by the captured object}}
       ^{ [x actNow]; }]; // expected-warning {{capturing 'x' strongly in this block is likely to lead to a retain cycle}}

  // These actually don't cause retain cycles.
  __weak Test0 *weakx = x;
  [x addBlock: ^{ [weakx actNow]; }];
  [x setBlock: ^{ [weakx actNow]; }];
  x.block = ^{ [weakx actNow]; };

  // These do cause retain cycles, but we're not clever enough to figure that out.
  [weakx addBlock: ^{ [x actNow]; }];
  [weakx setBlock: ^{ [x actNow]; }];
  weakx.block = ^{ [x actNow]; };

  // rdar://11702054
  x.block = ^{ (void)x.actNow; };  // expected-warning {{capturing 'x' strongly in this block is likely to lead to a retain cycle}} \
                                   // expected-note {{block will be retained by the captured object}}
}

@interface BlockOwner
@property (retain) void (^strong)(void); // expected-warning {{retain'ed block property does not copy the block - use copy attribute instead}}
@end

@interface Test1 {
@public
  BlockOwner *owner;
};
@property (retain) BlockOwner *owner;
@property (assign) __strong BlockOwner *owner2; // expected-error {{unsafe_unretained property 'owner2' may not also be declared __strong}}
@property (assign) BlockOwner *owner3;
@end
void test1(Test1 *x) {
  x->owner.strong = ^{ (void) x; }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}
  x.owner.strong = ^{ (void) x; }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}
  x.owner2.strong = ^{ (void) x; };
  x.owner3.strong = ^{ (void) x; };
}

@implementation Test1 {
  BlockOwner * __unsafe_unretained owner3ivar;
  __weak BlockOwner *weakowner;
}
@dynamic owner;
@dynamic owner2;
@synthesize owner3 = owner3ivar;

- (id) init {
  self.owner.strong = ^{ (void) owner; }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}
  self.owner2.strong = ^{ (void) owner; };

  // TODO: should we warn here?  What's the story with this kind of mismatch?
  self.owner3.strong = ^{ (void) owner; };

  owner.strong = ^{ (void) owner; }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}

  owner.strong = ^{ ^{ (void) owner; }(); }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}

  owner.strong = ^{ (void) sizeof(self); // expected-note {{block will be retained by an object strongly retained by the captured object}}
                    (void) owner; }; // expected-warning {{capturing 'self' strongly in this block is likely to lead to a retain cycle}}

  weakowner.strong = ^{ (void) owner; };

  return self;
}
- (void) foo {
  owner.strong = ^{ (void) owner; }; // expected-warning {{retain cycle}} expected-note {{block will be retained by an object strongly retained by the captured object}}
}
@end

void test2_helper(id);
@interface Test2 {
  void (^block)(void);
  id x;
}
@end
@implementation Test2
- (void) test {
  block = ^{ // expected-note {{block will be retained by an object strongly retained by the captured object}}
    test2_helper(x); // expected-warning {{capturing 'self' strongly in this block is likely to lead to a retain cycle}}
  };
}
@end


@interface NSOperationQueue {}
- (void)addOperationWithBlock:(void (^)(void))block;
- (void)addSomethingElse:(void (^)(void))block;

@end

@interface Test3 {
  NSOperationQueue *myOperationQueue;
  unsigned count;
}
@end
void doSomething(unsigned v);
@implementation Test3
- (void) test {
  // 'addOperationWithBlock:' is specifically whitelisted.
  [myOperationQueue addOperationWithBlock:^() { // no-warning
    if (count > 20) {
      doSomething(count);
    }
  }];
}
- (void) test_positive {
  // Sanity check that we are really whitelisting 'addOperationWithBlock:' and not doing
  // something funny.
  [myOperationQueue addSomethingElse:^() { // expected-note {{block will be retained by an object strongly retained by the captured object}}
    if (count > 20) { // expected-warning {{capturing 'self' strongly in this block is likely to lead to a retain cycle}}
      doSomething(count);
    }
  }];
}
@end


void testBlockVariable() {
  typedef void (^block_t)(void);
  
  // This case will be caught by -Wuninitialized, and does not create a
  // retain cycle.
  block_t a1 = ^{
    a1(); // no-warning
  };

  // This case will also be caught by -Wuninitialized.
  block_t a2;
  a2 = ^{
    a2(); // no-warning
  };
  
  __block block_t b1 = ^{ // expected-note{{block will be retained by the captured object}}
    b1(); // expected-warning{{capturing 'b1' strongly in this block is likely to lead to a retain cycle}}
  };

  __block block_t b2;
  b2 = ^{ // expected-note{{block will be retained by the captured object}}
    b2(); // expected-warning{{capturing 'b2' strongly in this block is likely to lead to a retain cycle}}
  };
}


@interface NSObject
- (id)copy;

- (void (^)(void))someRandomMethodReturningABlock;
@end


void testCopying(Test0 *obj) {
  typedef void (^block_t)(void);

  [obj setBlock:[^{ // expected-note{{block will be retained by the captured object}}
    [obj actNow]; // expected-warning{{capturing 'obj' strongly in this block is likely to lead to a retain cycle}}
  } copy]];

  [obj addBlock:(__bridge_transfer block_t)_Block_copy((__bridge void *)^{ // expected-note{{block will be retained by the captured object}}
    [obj actNow]; // expected-warning{{capturing 'obj' strongly in this block is likely to lead to a retain cycle}}
  })];
  
  [obj addBlock:[^{
    [obj actNow]; // no-warning
  } someRandomMethodReturningABlock]];
  
  extern block_t someRandomFunctionReturningABlock(block_t);
  [obj setBlock:someRandomFunctionReturningABlock(^{
    [obj actNow]; // no-warning
  })];
}

