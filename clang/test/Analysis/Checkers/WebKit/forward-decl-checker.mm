// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.ForwardDeclChecker -verify %s

#include "mock-types.h"
#include "objc-mock-types.h"
#include "mock-system-header.h"

namespace std {

template <typename T> struct remove_reference {
  typedef T type;
};

template <typename T> struct remove_reference<T&> {
  typedef T type;
};

template<typename T> typename remove_reference<T>::type&& move(T&& t);

} // namespace std

typedef struct OpaqueJSString * JSStringRef;

class Obj;
@class ObjCObj;

Obj* provide_obj_ptr();
void receive_obj_ptr(Obj* p = nullptr);

Obj* ptr(Obj* arg) {
  receive_obj_ptr(provide_obj_ptr());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj *'}}
  auto *obj = provide_obj_ptr();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj *'}}
  receive_obj_ptr(arg);
  receive_obj_ptr(nullptr);
  receive_obj_ptr();
  return obj;
}

Obj& provide_obj_ref();
void receive_obj_ref(Obj& p);

Obj& ref() {
  receive_obj_ref(provide_obj_ref());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj &'}}
  auto &obj = provide_obj_ref();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj &'}}
  return obj;
}

Obj&& provide_obj_rval();
void receive_obj_rval(Obj&& p);

void rval(Obj&& arg) {
  receive_obj_rval(provide_obj_rval());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj &&'}}
  auto &&obj = provide_obj_rval();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj &&'}}
  receive_obj_rval(std::move(arg));
}

ObjCObj *provide_objcobj();
void receive_objcobj(ObjCObj *p);
ObjCObj *objc_ptr() {
  receive_objcobj(provide_objcobj());
  auto *objcobj = provide_objcobj();
  return objcobj;
}

struct WrapperObj {
  Obj* ptr { nullptr };
  // expected-warning@-1{{Member variable 'ptr' uses a forward declared type 'Obj *'}}

  WrapperObj(Obj* obj);
  WrapperObj(Obj& obj);
  WrapperObj(Obj&& obj);
};

void construct_ptr(Obj&& arg) {
  WrapperObj wrapper1(provide_obj_ptr());
  // expected-warning@-1{{Call argument for parameter 'obj' uses a forward declared type 'Obj *'}}
  WrapperObj wrapper2(provide_obj_ref());
  // expected-warning@-1{{Call argument for parameter 'obj' uses a forward declared type 'Obj &'}}
  WrapperObj wrapper3(std::move(arg));
}

JSStringRef provide_opaque_ptr();
void receive_opaque_ptr(JSStringRef);
NSZone *provide_zone();

JSStringRef opaque_ptr() {
  receive_opaque_ptr(provide_opaque_ptr());
  auto ref = provide_opaque_ptr();
  return ref;
}

@interface AnotherObj : NSObject
- (Obj *)ptr;
- (Obj &)ref;
- (void)objc;
- (void)doMoreWork:(ObjCObj *)obj;
@end

@implementation AnotherObj
- (Obj *)ptr {
  receive_obj_ptr(provide_obj_ptr());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj *'}}
  auto *obj = provide_obj_ptr();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj *'}}
  return obj;
}

- (Obj &)ref {
  receive_obj_ref(provide_obj_ref());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj &'}}
  auto &obj = provide_obj_ref();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj &'}}
  return obj;
}

- (void)objc {
  auto *obj = provide_objcobj();
  [obj doWork];
  [self doMoreWork:provide_objcobj()];
  [self doMoreWork:nil];
}

- (void)doMoreWork:(ObjCObj *)obj {
  auto array = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  CFArrayAppendValue(array, nullptr);
  auto log = os_log_create("Foo", "Bar");
  os_log_msg(log, OS_LOG_TYPE_DEFAULT, "Some Log");
  auto *zone = provide_zone();
}

@end
