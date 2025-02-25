// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.RetainPtrCtorAdoptChecker -fobjc-arc -verify %s

#include "objc-mock-types.h"

void basic_correct() {
  auto ns1 = adoptNS([SomeObj alloc]);
  auto ns2 = adoptNS([[SomeObj alloc] init]);
  RetainPtr<SomeObj> ns3 = [ns1.get() next];
  auto ns4 = adoptNS([ns3 mutableCopy]);
  auto ns5 = adoptNS([ns3 copyWithValue:3]);
  auto ns6 = retainPtr([ns3 next]);
  CFMutableArrayRef cf1 = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 10));
}

CFMutableArrayRef provide_cf();

void basic_wrong() {
  RetainPtr<SomeObj> ns1 = [[SomeObj alloc] init];
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak when ARC is disabled [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  auto ns2 = adoptNS([ns1.get() next]);
  // expected-warning@-1{{Incorrect use of adoptNS. The argument is +0 and results in an use-after-free when ARC is disabled [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf1 = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf2 = adoptCF(provide_cf());
  // expected-warning@-1{{Incorrect use of adoptCF. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

RetainPtr<SomeObj> return_nullptr() {
  return nullptr;
}

RetainPtr<SomeObj> return_retainptr() {
  RetainPtr<SomeObj> foo;
  return foo;
}

CFTypeRef CopyValueForSomething();

void cast_retainptr() {
  RetainPtr<NSObject> foo;
  RetainPtr<SomeObj> bar = static_cast<SomeObj*>(foo);

  auto baz = adoptCF(CopyValueForSomething()).get();
  RetainPtr<CFArrayRef> v = static_cast<CFArrayRef>(baz);
}

void adopt_retainptr() {
  RetainPtr<NSObject> foo = adoptNS([[SomeObj alloc] init]);
}

RetainPtr<CFArrayRef> return_arg(CFArrayRef arg) {
  return arg;
}

class MemberInit {
public:
  MemberInit(CFMutableArrayRef array, NSString *str, CFRunLoopRef runLoop)
    : m_array(array)
    , m_str(str)
    , m_runLoop(runLoop)
  { }

private:
  RetainPtr<CFMutableArrayRef> m_array;
  RetainPtr<NSString> m_str;
  RetainPtr<CFRunLoopRef> m_runLoop;
};
void create_member_init() {
  MemberInit init { CFArrayCreateMutable(kCFAllocatorDefault, 10), @"hello", CFRunLoopGetCurrent() };
}

RetainPtr<CFStringRef> cfstr() {
  return CFSTR("");
}

template <typename CF, typename NS>
static RetainPtr<NS> bridge_cast(RetainPtr<CF>&& ptr)
{
  return adoptNS((__bridge NSArray *)(ptr.leakRef()));
}

RetainPtr<CFArrayRef> create_cf_array();
RetainPtr<id> return_bridge_cast() {
  return bridge_cast<CFArrayRef, NSArray>(create_cf_array());
}
