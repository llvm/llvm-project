// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedCallArgsChecker -verify %s

#include "objc-mock-types.h"

SomeObj *provide();
void consume_obj(SomeObj*);

CFMutableArrayRef provide_cf();
void consume_cf(CFMutableArrayRef);

dispatch_queue_t provide_dispatch();
void consume_dispatch(dispatch_queue_t);

CGImageRef provideImage();
NSString *stringForImage(CGImageRef);

void some_function();

namespace simple {
  void foo() {
    consume_obj(provide());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
    consume_cf(provide_cf());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
    consume_dispatch(provide_dispatch());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  }

  // Test that the checker works with [[clang::suppress]].
  void foo_suppressed() {
    [[clang::suppress]] consume_obj(provide()); // no-warning
    [[clang::suppress]] consume_cf(provide_cf()); // no-warning
  }

}

namespace multi_arg {
  void consume_retainable(int, SomeObj* foo, CFMutableArrayRef bar, dispatch_queue_t baz, bool);
  void foo() {
    consume_retainable(42, provide(), provide_cf(), provide_dispatch(), true);
    // expected-warning@-1{{Call argument for parameter 'foo' is unretained and unsafe}}
    // expected-warning@-2{{Call argument for parameter 'bar' is unretained and unsafe}}
    // expected-warning@-3{{Call argument for parameter 'baz' is unretained and unsafe}}
  }

  void consume_retainable(SomeObj* foo, ...);
  void bar() {
    consume_retainable(provide(), 1, provide_cf(), RetainPtr<CFMutableArrayRef> { provide_cf() }.get(), provide_dispatch());
    // expected-warning@-1{{Call argument for parameter 'foo' is unretained and unsafe}}
    // expected-warning@-2{{Call argument is unretained and unsafe}}
    // expected-warning@-3{{Call argument is unretained and unsafe}}
     consume_retainable(RetainPtr<SomeObj> { provide() }.get(), 1, RetainPtr<CFMutableArrayRef> { provide_cf() }.get());
  }
}

namespace retained {
  RetainPtr<SomeObj> provide_obj();
  RetainPtr<CFMutableArrayRef> provide_cf();
  OSObjectPtr<dispatch_queue_t> provide_dispatch();
  void foo() {
    consume_obj(provide_obj().get()); // no warning
    consume_cf(provide_cf().get()); // no warning
    consume_dispatch(provide_dispatch().get()); // no warning
  }
}

namespace methods {
  struct Consumer {
    void consume_obj(SomeObj* ptr);
    void consume_cf(CFMutableArrayRef ref);
    void consume_dispatch(dispatch_queue_t ptr);
  };

  void foo() {
    Consumer c;

    c.consume_obj(provide());
    // expected-warning@-1{{Call argument for parameter 'ptr' is unretained and unsafe}}
    c.consume_cf(provide_cf());
    // expected-warning@-1{{Call argument for parameter 'ref' is unretained and unsafe}}
    c.consume_dispatch(provide_dispatch());
    // expected-warning@-1{{Call argument for parameter 'ptr' is unretained and unsafe}}
  }

  void foo2() {
    struct Consumer {
      void consume(SomeObj*) { some_function(); }
      void whatever() {
        consume(provide());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }

      void consume_cf(CFMutableArrayRef) { some_function(); }
      void something() {
        consume_cf(provide_cf());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }

      void consume_dispatch(dispatch_queue_t) { some_function(); }
      void anything() {
        consume_dispatch(provide_dispatch());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }
    };
  }

  void foo3() {
    struct Consumer {
      void consume(SomeObj*) { some_function(); }
      void whatever() {
        this->consume(provide());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }

      void consume_cf(CFMutableArrayRef) { some_function(); }
      void something() {
        this->consume_cf(provide_cf());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }

      void consume_dispatch(dispatch_queue_t) { some_function(); }
      void anything() {
        this->consume_dispatch(provide_dispatch());
        // expected-warning@-1{{Call argument is unretained and unsafe}}
      }
    };
  }

}

namespace casts {
  void foo() {
    consume_obj(provide());
    // expected-warning@-1{{Call argument is unretained and unsafe}}

    consume_obj(static_cast<OtherObj*>(provide()));
    // expected-warning@-1{{Call argument is unretained and unsafe}}

    consume_obj(reinterpret_cast<OtherObj*>(provide()));
    // expected-warning@-1{{Call argument is unretained and unsafe}}

    consume_obj(downcast<OtherObj>(provide()));
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  }
}

namespace null_ptr {
  void foo_ref() {
    consume_obj(nullptr);
    consume_obj(0);
    consume_cf(nullptr);
    consume_cf(0);
    consume_dispatch(nullptr);
    consume_dispatch(0);
  }
}

namespace retain_ptr_lookalike {
  struct Decoy {
    SomeObj* get();
  };

  void foo() {
    Decoy D;

    consume_obj(D.get());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  }

  struct Decoy2 {
    CFMutableArrayRef get();
  };

  void bar() {
    Decoy2 D;

    consume_cf(D.get());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  }
}

namespace param_formarding_function {
  void consume_more_obj(OtherObj*);
  void consume_more_cf(CFMutableArrayRef);
  void consume_more_dispatch(dispatch_queue_t);

  namespace objc {
    void foo(SomeObj* param) {
      consume_more_obj(downcast<OtherObj>(param));
    }
  }

  namespace cf {
    void foo(CFMutableArrayRef param) {
      consume_more_cf(param);
    }
  }
}

namespace param_formarding_lambda {
  auto consume_more_obj = [](OtherObj*) { some_function(); };
  auto consume_more_cf = [](CFMutableArrayRef) { some_function(); };
  auto consume_more_dispatch = [](dispatch_queue_t) { some_function(); };

  namespace objc {
    void foo(SomeObj* param) {
      consume_more_obj(downcast<OtherObj>(param));
    }
  }

  namespace cf {
    void foo(CFMutableArrayRef param) {
      consume_more_cf(param);
    }
  }
  
  namespace os_obj {
    void foo(dispatch_queue_t param) {
      consume_more_dispatch(param);
    }
  }
}

namespace param_forwarding_method {
  struct Consumer {
    void consume_obj(SomeObj*);
    static void consume_obj_s(SomeObj*);
    void consume_cf(CFMutableArrayRef);
    static void consume_cf_s(CFMutableArrayRef);
    void consume_dispatch(dispatch_queue_t);
    static void consume_dispatch_s(dispatch_queue_t);
  };

  void bar(Consumer* consumer, SomeObj* param) {
    consumer->consume_obj(param);
  }

  void foo(SomeObj* param) {
    Consumer::consume_obj_s(param);
  }

  void baz(Consumer* consumer, CFMutableArrayRef param) {
    consumer->consume_cf(param);
    Consumer::consume_cf_s(param);
  }

  void baz(Consumer* consumer, dispatch_queue_t param) {
    consumer->consume_dispatch(param);
    Consumer::consume_dispatch_s(param);
  }
}


namespace default_arg {
  SomeObj* global;
  CFMutableArrayRef global_cf;
  dispatch_queue_t global_dispatch;

  void function_with_default_arg1(SomeObj* param = global);
  // expected-warning@-1{{Call argument for parameter 'param' is unretained and unsafe}}

  void function_with_default_arg2(CFMutableArrayRef param = global_cf);
  // expected-warning@-1{{Call argument for parameter 'param' is unretained and unsafe}}

  void function_with_default_arg3(dispatch_queue_t param = global_dispatch);
  // expected-warning@-1{{Call argument for parameter 'param' is unretained and unsafe}}

  void foo() {
    function_with_default_arg1();
    function_with_default_arg2();
    function_with_default_arg3();
  }
}

namespace cxx_member_func {
  RetainPtr<SomeObj> protectedProvide();
  RetainPtr<CFMutableArrayRef> protectedProvideCF();

  void foo() {
    [provide() doWork];
    // expected-warning@-1{{Receiver is unretained and unsafe}}
    [protectedProvide().get() doWork];

    CFArrayAppendValue(provide_cf(), nullptr);
    // expected-warning@-1{{Call argument for parameter 'theArray' is unretained and unsafe}}
    CFArrayAppendValue(protectedProvideCF(), nullptr);
  };

  void bar() {
    [downcast<OtherObj>(protectedProvide().get()) doMoreWork:downcast<OtherObj>(provide())];
    // expected-warning@-1{{Call argument for parameter 'other' is unretained and unsafe}}
    [protectedProvide().get() doWork];
  };

}

namespace cxx_member_operator_call {
  // The hidden this-pointer argument without a corresponding parameter caused couple bugs in parameter <-> argument attribution.
  struct Foo {
    Foo& operator+(SomeObj* bad);
    friend Foo& operator-(Foo& lhs, SomeObj* bad);
    void operator()(SomeObj* bad);
    Foo& operator<<(dispatch_queue_t bad);
  };

  SomeObj* global;
  dispatch_queue_t global_dispatch;

  void foo() {
    Foo f;
    f + global;
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
    f - global;
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
    f(global);
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
    f << global_dispatch;
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
  }
}

namespace cxx_assignment_op {

  SomeObj* provide();
  void foo() {
    RetainPtr<SomeObj> ptr;
    ptr = provide();
    OSObjectPtr<dispatch_queue_t> objPtr;
    objPtr = provide_dispatch();
  }

}

namespace call_with_ptr_on_ref {
  RetainPtr<SomeObj> provideProtected();
  RetainPtr<CFMutableArrayRef> provideProtectedCF();
  OSObjectPtr<dispatch_queue_t> provideProtectedDispatch();
  void bar(SomeObj* bad);
  void bar_cf(CFMutableArrayRef bad);
  void bar_dispatch(dispatch_queue_t);
  bool baz();
  void foo(bool v) {
    bar(v ? nullptr : provideProtected().get());
    bar(baz() ? provideProtected().get() : nullptr);
    bar(v ? provide() : provideProtected().get());
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
    bar(v ? provideProtected().get() : provide());
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}

    bar_cf(v ? nullptr : provideProtectedCF().get());
    bar_cf(baz() ? provideProtectedCF().get() : nullptr);
    bar_cf(v ? provide_cf() : provideProtectedCF().get());
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}
    bar_cf(v ? provideProtectedCF().get() : provide_cf());
    // expected-warning@-1{{Call argument for parameter 'bad' is unretained and unsafe}}

    bar_dispatch(v ? nullptr : provideProtectedDispatch().get());
    bar_dispatch(baz() ? provideProtectedDispatch().get() : nullptr);
    bar_dispatch(v ? provide_dispatch() : provideProtectedDispatch().get());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
    bar_dispatch(v ? provideProtectedDispatch().get() : provide_dispatch());
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  }
}

namespace call_with_explicit_temporary_obj {
  void foo() {
    [RetainPtr<SomeObj>(provide()).get() doWork];
    CFArrayAppendValue(RetainPtr<CFMutableArrayRef> { provide_cf() }.get(), nullptr);
  }
  template <typename T>
  void bar() {
    [RetainPtr<SomeObj>(provide()).get() doWork];
    CFArrayAppendValue(RetainPtr<CFMutableArrayRef> { provide_cf() }.get(), nullptr);
  }
  void baz() {
    bar<int>();
  }
  void os_ptr() {
    consume_dispatch(OSObjectPtr { provide_dispatch() }.get());
  }
}

namespace call_with_adopt_ref {
  void foo() {
    [adoptNS(provide()).get() doWork];
    CFArrayAppendValue(adoptCF(provide_cf()).get(), nullptr);
    consume_dispatch(adoptOSObject(provide_dispatch()).get());
  }
}

#define YES __objc_yes
#define NO 0

namespace call_with_cf_constant {
  void bar(const NSArray *);
  void baz(const NSDictionary *);
  void boo(NSNumber *);
  void boo(CFTypeRef);

  struct Details {
    int value;
  };

  void foo(Details* details) {
    CFArrayCreateMutable(kCFAllocatorDefault, 10);
    bar(@[@"hello"]);
    baz(@{@"hello": @3});
    boo(@YES);
    boo(@NO);
    boo(@(details->value));
  }
}

namespace call_with_cf_string {
  void bar(CFStringRef);
  void foo() {
    bar(CFSTR("hello"));
  }
}

namespace call_with_ns_string {
  void bar(NSString *);
  void foo() {
    bar(@"world");
  }
}

namespace bridge_cast_arg {
  void bar(NSString *);
  void baz(NSString *);
  extern const CFStringRef kCFBundleNameKey;

  NSObject *foo(CFStringRef arg) {
    bar((NSString *)bridge_cast((CFTypeRef)arg));
    auto dict = @{
      @"hello": @1,
    };
    return dict[(__bridge NSString *)kCFBundleNameKey];
  }
}

namespace alloc_init_pair {
  void foo() {
    auto obj = adoptNS([[SomeObj alloc] init]);
    [obj doWork];
  }
}

namespace alloc_class {
  bool foo(NSObject *obj) {
    return [obj isKindOfClass:SomeObj.class] && [obj isKindOfClass:NSClassFromString(@"SomeObj")];
  }

  bool bar(NSObject *obj) {
    return [obj isKindOfClass:[SomeObj class]];
  }

  bool baz(NSObject *obj) {
    return [obj isKindOfClass:[SomeObj superclass]];
  }
}

namespace ptr_conversion {

SomeObj *provide_obj();

void dobjc(SomeObj* obj) {
  [dynamic_objc_cast<OtherObj>(obj) doMoreWork:nil];
}

void cobjc(SomeObj* obj) {
  [checked_objc_cast<OtherObj>(obj) doMoreWork:nil];
}

unsigned dcf(CFTypeRef obj) {
  return CFArrayGetCount(dynamic_cf_cast<CFArrayRef>(obj));
}

unsigned ccf(CFTypeRef obj) {
  return CFArrayGetCount(checked_cf_cast<CFArrayRef>(obj));
}

void some_function(id);
void idcf(CFTypeRef obj) {
  some_function(bridge_id_cast(obj));
}

} // ptr_conversion

namespace const_global {

extern NSString * const SomeConstant;
extern CFDictionaryRef const SomeDictionary;
extern dispatch_queue_t const SomeDispatch;
void doWork(NSString *str, CFDictionaryRef dict, dispatch_queue_t dispatch);
void use_const_global() {
  doWork(SomeConstant, SomeDictionary, SomeDispatch);
}

NSString *provide_str();
CFDictionaryRef provide_dict();
void use_const_local() {
  doWork(provide_str(), provide_dict(), provide_dispatch());
  // expected-warning@-1{{Call argument for parameter 'str' is unretained and unsafe}}
  // expected-warning@-2{{Call argument for parameter 'dict' is unretained and unsafe}}
  // expected-warning@-3{{Call argument for parameter 'dispatch' is unretained and unsafe}}
}

} // namespace const_global

namespace var_decl_ref_singleton {

static Class initSomeObject() { return nil; }
static Class (*getSomeObjectClassSingleton)() = initSomeObject;

bool foo(NSString *obj) {
  return [obj isKindOfClass:getSomeObjectClassSingleton()];
}

class Bar {
public:
  Class someObject();
  static Class staticSomeObject();
};
typedef Class (Bar::*SomeObjectSingleton)();

bool bar(NSObject *obj, Bar *bar, SomeObjectSingleton someObjSingleton) {
  return [obj isKindOfClass:(bar->*someObjSingleton)()];
  // expected-warning@-1{{Call argument for parameter 'aClass' is unretained and unsafe}}
}

bool baz(NSObject *obj) {
  Class (*someObjectSingleton)() = Bar::staticSomeObject;
  return [obj isKindOfClass:someObjectSingleton()];
}

} // namespace var_decl_ref_singleton

namespace ns_retained_return_value {

NSString *provideNS() NS_RETURNS_RETAINED;
CFDictionaryRef provideCF() CF_RETURNS_RETAINED;
dispatch_queue_t provideDispatch() NS_RETURNS_RETAINED;
void consumeNS(NSString *);
void consumeCF(CFDictionaryRef);
void consumeDispatch(dispatch_queue_t);

void foo() {
  consumeNS(provideNS());
  consumeCF(provideCF());
  consumeDispatch(provideDispatch());
}

struct Base {
  NSString *provideStr() NS_RETURNS_RETAINED;
};

struct Derived : Base {
  void consumeStr(NSString *);

  void foo() {
    consumeStr(provideStr());
  }
};

} // namespace ns_retained_return_value

@interface TestObject : NSObject
- (void)doWork:(NSString *)msg, ...;
- (void)doWorkOnSelf;
- (SomeObj *)getSomeObj;
@end

@implementation TestObject

- (void)doWork:(NSString *)msg, ... {
  some_function();
}

- (void)doWorkOnSelf {
  [self doWork:nil];
  [self doWork:@"hello", provide(), provide_cf(), provide_dispatch()];
  // expected-warning@-1{{Call argument is unretained and unsafe}}
  // expected-warning@-2{{Call argument is unretained and unsafe}}
  // expected-warning@-3{{Call argument is unretained and unsafe}}
  [self doWork:@"hello", RetainPtr<SomeObj> { provide() }.get(), RetainPtr<CFMutableArrayRef> { provide_cf() }.get(), OSObjectPtr { provide_dispatch() }.get()];
  [self doWork:__null];
  [self doWork:nil];
  [NSApp run];
}

- (SomeObj *)getSomeObj {
    return RetainPtr<SomeObj *>(provide()).autorelease();
}

- (void)doWorkOnSomeObj {
    [[self getSomeObj] doWork];
}

- (CGImageRef)createImage {
  return provideImage();
}

- (NSString *)convertImage {
  RetainPtr<CGImageRef> image = [self createImage];
  return stringForImage(image.get());
}
@end
