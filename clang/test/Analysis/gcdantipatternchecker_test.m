// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.performance.GCDAntipattern %s -fblocks -verify
typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end

typedef int dispatch_semaphore_t;
typedef void (^block_t)();

dispatch_semaphore_t dispatch_semaphore_create(int);
void dispatch_semaphore_wait(dispatch_semaphore_t, int);
void dispatch_semaphore_signal(dispatch_semaphore_t);

void func(void (^)(void));
void func_w_typedef(block_t);

int coin();

void use_semaphor_antipattern() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

// It's OK to use pattern in tests.
// We simply match the containing function name against ^test.
void test_no_warning() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100);
}

void use_semaphor_antipattern_multiple_times() {
  dispatch_semaphore_t sema1 = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema1);
  });
  dispatch_semaphore_wait(sema1, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}

  dispatch_semaphore_t sema2 = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema2);
  });
  dispatch_semaphore_wait(sema2, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void use_semaphor_antipattern_multiple_wait() {
  dispatch_semaphore_t sema1 = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema1);
  });
  // FIXME: multiple waits on same semaphor should not raise a warning.
  dispatch_semaphore_wait(sema1, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
  dispatch_semaphore_wait(sema1, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void warn_incorrect_order() {
  // FIXME: ASTMatchers do not allow ordered matching, so would match even
  // if out of order.
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
  func(^{
      dispatch_semaphore_signal(sema);
  });
}

void warn_w_typedef() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func_w_typedef(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void warn_nested_ast() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  if (coin()) {
    func(^{
         dispatch_semaphore_signal(sema);
         });
  } else {
    func(^{
         dispatch_semaphore_signal(sema);
         });
  }
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void use_semaphore_assignment() {
  dispatch_semaphore_t sema;
  sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void use_semaphore_assignment_init() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  sema = dispatch_semaphore_create(1);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void differentsemaphoreok() {
  dispatch_semaphore_t sema1 = dispatch_semaphore_create(0);
  dispatch_semaphore_t sema2 = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema1);
  });
  dispatch_semaphore_wait(sema2, 100); // no-warning
}

void nosignalok() {
  dispatch_semaphore_t sema1 = dispatch_semaphore_create(0);
  dispatch_semaphore_wait(sema1, 100);
}

void nowaitok() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  func(^{
      dispatch_semaphore_signal(sema);
  });
}

void noblockok() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  dispatch_semaphore_signal(sema);
  dispatch_semaphore_wait(sema, 100);
}

void storedblockok() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);
  block_t b = ^{
      dispatch_semaphore_signal(sema);
  };
  dispatch_semaphore_wait(sema, 100);
}

void passed_semaphore_ok(dispatch_semaphore_t sema) {
  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100);
}

void warn_with_cast() {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal((int)sema);
  });
  dispatch_semaphore_wait((int)sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

@interface MyInterface1 : NSObject
-(void)use_method_warn;
-(void) pass_block_as_second_param_warn;
-(void)use_objc_callback_warn;
-(void)testNoWarn;
-(void)acceptBlock:(block_t)callback;
-(void)flag:(int)flag acceptBlock:(block_t)callback;
@end

@implementation MyInterface1

-(void)use_method_warn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

-(void) pass_block_as_second_param_warn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  [self flag:1 acceptBlock:^{
      dispatch_semaphore_signal(sema);
  }];
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

-(void)testNoWarn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100);
}

-(void)acceptBlock:(block_t) callback {
  callback();
}

-(void)flag:(int)flag acceptBlock:(block_t)callback {
  callback();
}

-(void)use_objc_callback_warn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  [self acceptBlock:^{
      dispatch_semaphore_signal(sema);
  }];
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}

void use_objc_and_c_callback(MyInterface1 *t) {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}

  dispatch_semaphore_t sema1 = dispatch_semaphore_create(0);

  [t acceptBlock:^{
      dispatch_semaphore_signal(sema1);
  }];
  dispatch_semaphore_wait(sema1, 100); // expected-warning{{Waiting on a semaphore with Grand Central Dispatch creates useless threads and is subject to priority inversion}}
}
@end

// No warnings: class name contains "test"
@interface Test1 : NSObject
-(void)use_method_warn;
@end

@implementation Test1
-(void)use_method_warn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100);
}
@end


// No warnings: class name contains "mock"
@interface Mock1 : NSObject
-(void)use_method_warn;
@end

@implementation Mock1
-(void)use_method_warn {
  dispatch_semaphore_t sema = dispatch_semaphore_create(0);

  func(^{
      dispatch_semaphore_signal(sema);
  });
  dispatch_semaphore_wait(sema, 100);
}
@end
