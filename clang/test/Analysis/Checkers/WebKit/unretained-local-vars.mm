// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedLocalVarsChecker -verify %s

#import "objc-mock-types.h"

void someFunction();

namespace raw_ptr {
void foo() {
  SomeObj *bar;
  // FIXME: later on we might warn on uninitialized vars too
}

void bar(SomeObj *) {}
} // namespace raw_ptr

namespace pointer {
SomeObj *provide();
void foo_ref() {
  SomeObj *bar = provide();
  // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  [bar doWork];
}

bool bar_ref(SomeObj *obj) {
    return !!obj;
}

void cf_ptr() {
  CFMutableArrayRef array = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Local variable 'array' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  CFArrayAppendValue(array, nullptr);
}
} // namespace pointer

namespace guardian_scopes {
void foo1() {
  RetainPtr<SomeObj> foo;
  {
    SomeObj *bar = foo.get();
  }
}

void foo2() {
  RetainPtr<SomeObj> foo;
  // missing embedded scope here
  SomeObj *bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  [bar doWork];
}

void foo3() {
  RetainPtr<SomeObj> foo;
  {
    { SomeObj *bar = foo.get(); }
  }
}

void foo4() {
  {
    RetainPtr<SomeObj> foo;
    { SomeObj *bar = foo.get(); }
  }
}

struct SelfReferencingStruct {
  SelfReferencingStruct* ptr;
  SomeObj* obj { nullptr };
};

void foo7(SomeObj* obj) {
  SelfReferencingStruct bar = { &bar, obj };
  [bar.obj doWork];
}

void foo8(SomeObj* obj) {
  RetainPtr<SomeObj> foo;

  {
    SomeObj *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    foo = nullptr;
    [bar doWork];
  }
  RetainPtr<SomeObj> baz;
  {
    SomeObj *bar = baz.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    baz = obj;
    [bar doWork];
  }

  foo = nullptr;
  {
    SomeObj *bar = foo.get();
    // No warning. It's okay to mutate RefPtr in an outer scope.
    [bar doWork];
  }
  foo = obj;
  {
    SomeObj *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    foo.clear();
    [bar doWork];
  }
  {
    SomeObj *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    foo = obj ? obj : nullptr;
    [bar doWork];
  }
  {
    SomeObj *bar = [foo.get() other] ? foo.get() : nullptr;
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    foo = nullptr;
    [bar doWork];
  }
}

void foo9(SomeObj* o) {
  RetainPtr<SomeObj> guardian(o);
  {
    SomeObj *bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    guardian = o; // We don't detect that we're setting it to the same value.
    [bar doWork];
  }
  {
    SomeObj *bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    RetainPtr<SomeObj> other(bar); // We don't detect other has the same value as guardian.
    guardian.swap(other);
    [bar doWork];
  }
  {
    SomeObj *bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    RetainPtr<SomeObj> other(static_cast<RetainPtr<SomeObj>&&>(guardian));
    [bar doWork];
  }
  {
    SomeObj *bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    guardian.clear();
    [bar doWork];
  }
  {
    SomeObj *bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    guardian = [o other] ? o : bar;
    [bar doWork];
  }
}

bool trivialFunction(CFMutableArrayRef array) { return !!array; }
void foo10() {
  RetainPtr<CFMutableArrayRef> array = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 10));
  {
    CFMutableArrayRef arrayRef = array.get();
    CFArrayAppendValue(arrayRef, nullptr);
  }
  {
    CFMutableArrayRef arrayRef = array.get();
    // expected-warning@-1{{Local variable 'arrayRef' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    array = nullptr;
    CFArrayAppendValue(arrayRef, nullptr);
  }
  {
    CFMutableArrayRef arrayRef = array.get();
    if (trivialFunction(arrayRef))
      arrayRef = nullptr;
  }
}

} // namespace guardian_scopes

namespace auto_keyword {
class Foo {
  SomeObj *provide_obj();
  CFMutableArrayRef provide_cf_array();
  void doWork(CFMutableArrayRef);

  void evil_func() {
    SomeObj *bar = provide_obj();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    auto *baz = provide_obj();
    // expected-warning@-1{{Local variable 'baz' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    auto *baz2 = this->provide_obj();
    // expected-warning@-1{{Local variable 'baz2' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    [[clang::suppress]] auto *baz_suppressed = provide_obj(); // no-warning
  }

  void func() {
    SomeObj *bar = provide_obj();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    if (bar)
      [bar doWork];
  }

  void bar() {
    auto bar = provide_cf_array();
    // expected-warning@-1{{Local variable 'bar' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    doWork(bar);
    [[clang::suppress]] auto baz = provide_cf_array(); // no-warning
    doWork(baz);
  }

};
} // namespace auto_keyword

namespace guardian_casts {
void foo1() {
  RetainPtr<NSObject> foo;
  {
    SomeObj *bar = downcast<SomeObj>(foo.get());
    [bar doWork];
  }
}

void foo2() {
  RetainPtr<NSObject> foo;
  {
    SomeObj *bar = static_cast<SomeObj *>(downcast<SomeObj>(foo.get()));
    someFunction();
  }
}
} // namespace guardian_casts

namespace conditional_op {
SomeObj *provide_obj();
bool bar();

void foo() {
  SomeObj *a = bar() ? nullptr : provide_obj();
  // expected-warning@-1{{Local variable 'a' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  RetainPtr<SomeObj> b = provide_obj();
  {
    SomeObj* c = bar() ? nullptr : b.get();
    [c doWork];
    SomeObj* d = bar() ? b.get() : nullptr;
    [d doWork];
  }
}

} // namespace conditional_op

namespace local_assignment_basic {

SomeObj *provide_obj();

void foo(SomeObj* a) {
  SomeObj* b = a;
  // expected-warning@-1{{Local variable 'b' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  if ([b other])
    b = provide_obj();
}

void bar(SomeObj* a) {
  SomeObj* b;
  // expected-warning@-1{{Local variable 'b' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  b = provide_obj();
}

void baz() {
  RetainPtr<SomeObj> a = provide_obj();
  {
    SomeObj* b = a.get();
    // expected-warning@-1{{Local variable 'b' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
    b = provide_obj();
  }
}

} // namespace local_assignment_basic

namespace local_assignment_to_parameter {

SomeObj *provide_obj();
void someFunction();

void foo(SomeObj* a) {
  a = provide_obj();
  // expected-warning@-1{{Assignment to an unretained parameter 'a' is unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  someFunction();
  [a doWork];
}

CFMutableArrayRef provide_cf_array();
void doWork(CFMutableArrayRef);

void bar(CFMutableArrayRef a) {
  a = provide_cf_array();
  // expected-warning@-1{{Assignment to an unretained parameter 'a' is unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  doWork(a);
}

} // namespace local_assignment_to_parameter

namespace local_assignment_to_static_local {

SomeObj *provide_obj();
void someFunction();

void foo() {
  static SomeObj* a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  a = provide_obj();
  someFunction();
  [a doWork];
}

CFMutableArrayRef provide_cf_array();
void doWork(CFMutableArrayRef);

void bar() {
  static CFMutableArrayRef a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  a = provide_cf_array();
  doWork(a);
}

} // namespace local_assignment_to_static_local

namespace local_assignment_to_global {

SomeObj *provide_obj();
void someFunction();

SomeObj* g_a = nullptr;
// expected-warning@-1{{Global variable 'local_assignment_to_global::g_a' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}

void foo() {
  g_a = provide_obj();
  someFunction();
  [g_a doWork];
}

CFMutableArrayRef provide_cf_array();
void doWork(CFMutableArrayRef);

CFMutableArrayRef g_b = nullptr;
// expected-warning@-1{{Global variable 'local_assignment_to_global::g_b' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}

void bar() {
  g_b = provide_cf_array();
  doWork(g_b);
}

} // namespace local_assignment_to_global

namespace local_var_for_singleton {
  SomeObj *singleton();
  SomeObj *otherSingleton();
  void foo() {
    SomeObj* bar = singleton();
    SomeObj* baz = otherSingleton();
  }

  CFMutableArrayRef cfSingleton();
  void bar() {
    CFMutableArrayRef cf = cfSingleton();
  }
}

namespace ptr_conversion {

SomeObj *provide_obj();

void dobjc(SomeObj* obj) {
  if (auto *otherObj = dynamic_objc_cast<OtherObj>(obj))
    [otherObj doMoreWork:nil];
}

void cobjc(SomeObj* obj) {
  auto *otherObj = checked_objc_cast<OtherObj>(obj);
  [otherObj doMoreWork:nil];
}

unsigned dcf(CFTypeRef obj) {
  if (CFArrayRef array = dynamic_cf_cast<CFArrayRef>(obj))
    return CFArrayGetCount(array);
  return 0;
}

unsigned ccf(CFTypeRef obj) {
  CFArrayRef array = checked_cf_cast<CFArrayRef>(obj);
  return CFArrayGetCount(array);
}

} // ptr_conversion

namespace const_global {

extern NSString * const SomeConstant;
extern CFDictionaryRef const SomeDictionary;
void doWork(NSString *, CFDictionaryRef);
void use_const_global() {
  doWork(SomeConstant, SomeDictionary);
}

NSString *provide_str();
CFDictionaryRef provide_dict();
void use_const_local() {
  NSString * const str = provide_str();
  // expected-warning@-1{{Local variable 'str' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  CFDictionaryRef dict = provide_dict();
  // expected-warning@-1{{Local variable 'dict' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  doWork(str, dict);
}

} // namespace const_global

namespace ns_retained_return_value {

NSString *provideNS() NS_RETURNS_RETAINED;
CFDictionaryRef provideCF() CF_RETURNS_RETAINED;
void consumeNS(NSString *);
void consumeCF(CFDictionaryRef);

unsigned foo() {
  auto *string = provideNS();
  auto *dictionary = provideCF();
  return string.length + CFDictionaryGetCount(dictionary);
}

} // namespace ns_retained_return_value

bool doMoreWorkOpaque(OtherObj*);
SomeObj* provide();

@implementation OtherObj
- (instancetype)init {
  self = [super init];
  return self;
}

- (void)doMoreWork:(OtherObj *)other {
  doMoreWorkOpaque(other);
}

- (SomeObj*)getSomeObj {
  return RetainPtr<SomeObj *>(provide()).autorelease();
}

- (void)storeSomeObj {
  auto *obj = [self getSomeObj];
  [obj doWork];
}
@end
