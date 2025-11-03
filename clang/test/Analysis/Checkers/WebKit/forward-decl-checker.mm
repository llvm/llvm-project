// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.ForwardDeclChecker -verify %s

#include "mock-types.h"
#include "objc-mock-types.h"
#include "mock-system-header.h"

typedef struct OpaqueJSString * JSStringRef;

class Obj;
@class ObjCObj;

Obj* provide_obj_ptr();
void receive_obj_ptr(Obj* p = nullptr);
void receive_obj_ref(Obj&);
void receive_obj_rref(Obj&&);
sqlite3* open_db();
void close_db(sqlite3*);

Obj* ptr(Obj* arg) {
  receive_obj_ptr(provide_obj_ptr());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj *'}}
  auto *obj = provide_obj_ptr();
  // expected-warning@-1{{Local variable 'obj' uses a forward declared type 'Obj *'}}
  receive_obj_ptr(arg);
  receive_obj_ptr(nullptr);
  receive_obj_ptr();
  auto* db = open_db();
  close_db(db);
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

void opaque_call_arg(Obj* obj, Obj&& otherObj, const RefPtr<Obj>& safeObj, WeakPtr<Obj> weakObj, std::unique_ptr<Obj>& uniqObj) {
  receive_obj_ref(*obj);
  receive_obj_ptr(&*obj);
  receive_obj_rref(std::move(otherObj));
  receive_obj_ref(*safeObj.get());
  receive_obj_ptr(weakObj.get());
  // expected-warning@-1{{Call argument for parameter 'p' uses a forward declared type 'Obj *'}}
  receive_obj_ref(*uniqObj);
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

namespace template_forward_declare {

template<typename> class HashSet;

template<typename T>
using SingleThreadHashSet = HashSet<T>;

template<typename> class HashSet { };

struct Font { };

struct ComplexTextController {
    SingleThreadHashSet<const Font>* fallbackFonts { nullptr };
};

}
