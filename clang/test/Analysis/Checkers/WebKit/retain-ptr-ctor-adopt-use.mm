// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.RetainPtrCtorAdoptChecker -verify %s

#include "objc-mock-types.h"

CFTypeRef CFCopyArray(CFArrayRef);
void* CreateCopy();

void basic_correct() {
  auto ns1 = adoptNS([SomeObj alloc]);
  auto ns2 = adoptNS([[SomeObj alloc] init]);
  RetainPtr<SomeObj> ns3 = [ns1.get() next];
  auto ns4 = adoptNS([ns3 mutableCopy]);
  auto ns5 = adoptNS([ns3 copyWithValue:3]);
  auto ns6 = retainPtr([ns3 next]);
  auto ns7 = retainPtr((SomeObj *)0);
  auto ns8 = adoptNS(nil);
  CFMutableArrayRef cf1 = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 10));
  auto cf2 = adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault));
  auto cf3 = adoptCF(checked_cf_cast<CFArrayRef>(CFCopyArray(cf1)));
  CreateCopy();
}

CFMutableArrayRef provide_cf();

void basic_wrong() {
  RetainPtr<SomeObj> ns1 = [[SomeObj alloc] init];
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  auto ns2 = adoptNS([ns1.get() next]);
  // expected-warning@-1{{Incorrect use of adoptNS. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf1 = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf2 = adoptCF(provide_cf());
  // expected-warning@-1{{Incorrect use of adoptCF. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFTypeRef> cf3 = SecTaskCreateFromSelf(kCFAllocatorDefault);
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  CFCopyArray(cf1);
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

void basic_correct_arc() {
  auto *obj = [[SomeObj alloc] init];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  [obj doWork];
}

@implementation SomeObj {
  NSNumber *_number;
  SomeObj *_next;
  SomeObj *_other;
}

- (instancetype)_init {
  self = [super init];
  _number = nil;
  _next = nil;
  _other = nil;
  return self;
}

- (SomeObj *)mutableCopy {
  auto *copy = [[SomeObj alloc] init];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  [copy setValue:_number];
  [copy setNext:_next];
  [copy setOther:_other];
  return copy;
}

- (SomeObj *)copyWithValue:(int)value {
  auto *copy = [[SomeObj alloc] init];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  [copy setValue:_number];
  [copy setNext:_next];
  [copy setOther:_other];
  return copy;
}

- (void)doWork {
  _number = [[NSNumber alloc] initWithInt:5];
}

- (SomeObj *)other {
  return _other;
}

- (void)setOther:(SomeObj *)obj {
  _other = obj;
}

- (SomeObj *)next {
  return _next;
}

- (void)setNext:(SomeObj *)obj {
  _next = obj;
}

- (int)value {
  return [_number intValue];
}

- (void)setValue:value {
  _number = value;
}

@end;

RetainPtr<CVPixelBufferRef> cf_out_argument() {
  auto surface = adoptCF(IOSurfaceCreate(nullptr));
  CVPixelBufferRef rawBuffer = nullptr;
  auto status = CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface.get(), nullptr, &rawBuffer);
  return adoptCF(rawBuffer);
}

RetainPtr<SomeObj> return_nil() {
  return nil;
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

CFTypeRef CopyWrapper() {
  return CopyValueForSomething();
}

CFTypeRef LeakWrapper() {
  return CopyValueForSomething();
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

NSArray *makeArray() NS_RETURNS_RETAINED {
  return (__bridge NSArray *)CFArrayCreateMutable(kCFAllocatorDefault, 10);
}

extern Class (*getNSArrayClass)();
NSArray *allocArrayInstance() NS_RETURNS_RETAINED {
  return [[getNSArrayClass() alloc] init];
}

extern int (*GetObj)(CF_RETURNS_RETAINED CFTypeRef* objOut);
RetainPtr<CFTypeRef> getObject() {
  CFTypeRef obj = nullptr;
  if (GetObj(&obj))
    return nullptr;
  return adoptCF(obj);
}

CFArrayRef CreateSingleArray(CFStringRef);
CFArrayRef CreateSingleArray(CFDictionaryRef);
CFArrayRef CreateSingleArray(CFArrayRef);
template <typename ElementType>
static RetainPtr<CFArrayRef> makeArrayWithSingleEntry(ElementType arg) {
  return adoptCF(CreateSingleArray(arg));
}

void callMakeArayWithSingleEntry() {
  auto dictionary = adoptCF(CFDictionaryCreate(kCFAllocatorDefault, nullptr, nullptr, 0));
  makeArrayWithSingleEntry(dictionary.get());
}

SomeObj* allocSomeObj() CF_RETURNS_RETAINED;

void adopt_retainptr() {
  RetainPtr<NSObject> foo = adoptNS([[SomeObj alloc] init]);
  auto bar = adoptNS([allocSomeObj() init]);
}

RetainPtr<CFArrayRef> return_arg(CFArrayRef arg) {
  return arg;
}

class MemberInit {
public:
  MemberInit(RetainPtr<CFMutableArrayRef>&& array, NSString *str, CFRunLoopRef runLoop)
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
  MemberInit init { adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 10)), @"hello", CFRunLoopGetCurrent() };
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

void mutable_copy_dictionary() {
  RetainPtr<NSMutableDictionary> mutableDictionary = adoptNS(@{
    @"Content-Type": @"text/html",
  }.mutableCopy);
}

void mutable_copy_array() {
  RetainPtr<NSMutableArray> mutableArray = adoptNS(@[
      @"foo",
  ].mutableCopy);
}

void string_copy(NSString *str) {
  RetainPtr<NSString> copy = adoptNS(str.copy);
}

void alloc_init_spi() {
  auto ptr = adoptNS([[SomeObj alloc] _init]);
}

void alloc_init_c_function() {
  RetainPtr ptr = adoptNS([allocSomeObj() init]);
}

void alloc_init_autorelease() {
  [[[SomeObj alloc] init] autorelease];
}

CFArrayRef make_array() CF_RETURNS_RETAINED;

RetainPtr<CFArrayRef> adopt_make_array() {
  return adoptCF(make_array());
}

@interface SomeObject : NSObject
-(void)basic_correct;
-(void)basic_wrong;
-(NSString *)leak_string;
-(NSString *)make_string NS_RETURNS_RETAINED;
@property (nonatomic, readonly) SomeObj *obj;
@end

@implementation SomeObject
-(void)basic_correct {
  auto ns1 = adoptNS([SomeObj alloc]);
  auto ns2 = adoptNS([[SomeObj alloc] init]);
  RetainPtr<SomeObj> ns3 = [ns1.get() next];
  auto ns4 = adoptNS([ns3 mutableCopy]);
  auto ns5 = adoptNS([ns3 copyWithValue:3]);
  auto ns6 = retainPtr([ns3 next]);
  CFMutableArrayRef cf1 = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 10));
  auto cf2 = adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault));
  auto cf3 = adoptCF(checked_cf_cast<CFArrayRef>(CFCopyArray(cf1)));
}

-(void)basic_wrong {
  RetainPtr<SomeObj> ns1 = [[SomeObj alloc] init];
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  auto ns2 = adoptNS([ns1.get() next]);
  // expected-warning@-1{{Incorrect use of adoptNS. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf1 = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFMutableArrayRef> cf2 = adoptCF(provide_cf());
  // expected-warning@-1{{Incorrect use of adoptCF. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  RetainPtr<CFTypeRef> cf3 = SecTaskCreateFromSelf(kCFAllocatorDefault);
  // expected-warning@-1{{Incorrect use of RetainPtr constructor. The argument is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  CFCopyArray(cf1);
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

-(NSString *)leak_string {
  return [[NSString alloc] initWithUTF8String:"hello"];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

-(NSString *)make_string {
  return [[NSString alloc] initWithUTF8String:"hello"];
}

-(void)local_leak_string {
  if ([[NSString alloc] initWithUTF8String:"hello"]) {
    // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  }
}

-(void)make_some_obj {
  auto some_obj = adoptNS([allocSomeObj() init]);
}

-(void)alloc_init_bad_order {
  auto *obj = [NSObject alloc];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
  auto ptr = adoptNS([obj init]);
  // expected-warning@-1{{Incorrect use of adoptNS. The argument is +0 and results in an use-after-free [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}

-(void)alloc_init_good_order {
  auto obj = adoptNS([NSObject alloc]);
  (void)[obj init];
}

-(void)copy_assign_ivar {
  _obj = [allocSomeObj() init];
}

-(void)do_more_work:(OtherObj *)otherObj {
  [otherObj doMoreWork:[[OtherObj alloc] init]];
  // expected-warning@-1{{The return value is +1 and results in a memory leak [alpha.webkit.RetainPtrCtorAdoptChecker]}}
}
@end
