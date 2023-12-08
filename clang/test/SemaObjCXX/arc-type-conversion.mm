// RUN: %clang_cc1 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify -fblocks %s

void * cvt(id arg) // expected-note{{candidate function not viable: cannot convert argument of incomplete type 'void *' to '__strong id'}}
{
  void* voidp_val;
  (void)(int*)arg; // expected-error {{cast of an Objective-C pointer to 'int *' is disallowed with ARC}}
  (void)(id)arg;
  (void)(__autoreleasing id*)arg; // expected-error{{cast of an Objective-C pointer to '__autoreleasing id *' is disallowed with ARC}}
  (void)(id*)arg; // expected-error{{cast of an Objective-C pointer to '__strong id *' is disallowed with ARC}}

  (void)(__autoreleasing id**)voidp_val;
  (void)(void*)voidp_val;
  (void)(void**)arg; // expected-error {{cast of an Objective-C pointer to 'void **' is disallowed}}
  cvt((void*)arg); // expected-error {{no matching function for call to 'cvt'}}
  cvt(0);
  (void)(__strong id**)(0);

  // FIXME: Diagnostic could be better here.
  return arg; // expected-error{{cannot initialize return object of type 'void *' with an lvalue of type '__strong id'}}
}

namespace rdar8898937 {

typedef void (^dispatch_block_t)(void);

void dispatch_once(dispatch_block_t block);
static void _dispatch_once(dispatch_block_t block)
{
  dispatch_once(block);
}

}

void static_casts(id arg) {
  void* voidp_val;
  (void)static_cast<int*>(arg); // expected-error {{cannot cast from type 'id' to pointer type 'int *'}}
  (void)static_cast<id>(arg);
  (void)static_cast<__autoreleasing id*>(arg); // expected-error{{cannot cast from type 'id' to pointer type '__autoreleasing id *'}}
  (void)static_cast<id*>(arg); // expected-error {{cannot cast from type 'id' to pointer type '__strong id *'}}

  (void)static_cast<__autoreleasing id**>(voidp_val);
  (void)static_cast<void*>(voidp_val);
  (void)static_cast<void**>(arg); // expected-error {{cannot cast from type 'id' to pointer type 'void **'}}
  (void)static_cast<__strong id**>(0);  

  __strong id *idp;
  (void)static_cast<__autoreleasing id*>(idp); // expected-error{{static_cast from '__strong id *' to '__autoreleasing id *' is not allowed}}
  (void)static_cast<__weak id*>(idp); // expected-error{{static_cast from '__strong id *' to '__weak id *' is not allowed}}
}

void test_const_cast(__strong id *sip, __weak id *wip, 
                     const __strong id *csip, __weak const id *cwip) {
  // Cannot use const_cast to cast between ownership qualifications or
  // add/remove ownership qualifications.
  (void)const_cast<__strong id *>(wip); // expected-error{{is not allowed}}
  (void)const_cast<__weak id *>(sip); // expected-error{{is not allowed}}

  // It's acceptable to cast away constness.
  (void)const_cast<__strong id *>(csip);
  (void)const_cast<__weak id *>(cwip);
}

void test_reinterpret_cast(__strong id *sip, __weak id *wip, 
                           const __strong id *csip, __weak const id *cwip) {
  // Okay to reinterpret_cast to add/remove/change ownership
  // qualifications.
  (void)reinterpret_cast<__strong id *>(wip);
  (void)reinterpret_cast<__weak id *>(sip);

  // Not allowed to cast away constness
  (void)reinterpret_cast<__strong id *>(csip); // expected-error{{reinterpret_cast from '__strong id const *' to '__strong id *' casts away qualifiers}}
  (void)reinterpret_cast<__weak id *>(cwip); // expected-error{{reinterpret_cast from '__weak id const *' to '__weak id *' casts away qualifiers}}
  (void)reinterpret_cast<__weak id *>(csip); // expected-error{{reinterpret_cast from '__strong id const *' to '__weak id *' casts away qualifiers}}
  (void)reinterpret_cast<__strong id *>(cwip); // expected-error{{reinterpret_cast from '__weak id const *' to '__strong id *' casts away qualifiers}}
}

void test_cstyle_cast(__strong id *sip, __weak id *wip, 
                      const __strong id *csip, __weak const id *cwip) {
  // C-style casts aren't allowed to change Objective-C ownership
  // qualifiers (beyond what the normal implicit conversion allows).

  (void)(__strong id *)wip; // expected-error{{C-style cast from '__weak id *' to '__strong id *' casts away qualifiers}}
  (void)(__strong id *)cwip; // expected-error{{C-style cast from '__weak id const *' to '__strong id *' casts away qualifiers}}
  (void)(__weak id *)sip; // expected-error{{C-style cast from '__strong id *' to '__weak id *' casts away qualifiers}}
  (void)(__weak id *)csip; // expected-error{{C-style cast from '__strong id const *' to '__weak id *' casts away qualifiers}}

  (void)(__strong const id *)wip; // expected-error{{C-style cast from '__weak id *' to '__strong id const *' casts away qualifiers}}
  (void)(__strong const id *)cwip; // expected-error{{C-style cast from '__weak id const *' to '__strong id const *' casts away qualifiers}}
  (void)(__weak const id *)sip; // expected-error{{C-style cast from '__strong id *' to '__weak id const *' casts away qualifiers}}
  (void)(__weak const id *)csip; // expected-error{{C-style cast from '__strong id const *' to '__weak id const *' casts away qualifiers}}
  (void)(__autoreleasing const id *)wip; // expected-error{{C-style cast from '__weak id *' to '__autoreleasing id const *' casts away qualifiers}}
  (void)(__autoreleasing const id *)cwip; // expected-error{{C-style cast from '__weak id const *' to '__autoreleasing id const *' casts away qualifiers}}
  (void)(__autoreleasing const id *)sip;
  (void)(__autoreleasing const id *)csip;
}

void test_functional_cast(__strong id *sip, __weak id *wip,
                          __autoreleasing id *aip) {
  // Functional casts aren't allowed to change Objective-C ownership
  // qualifiers (beyond what the normal implicit conversion allows).

  typedef __strong id *strong_id_pointer;
  typedef __weak id *weak_id_pointer;
  typedef __autoreleasing id *autoreleasing_id_pointer;

  typedef const __strong id *const_strong_id_pointer;
  typedef const __weak id *const_weak_id_pointer;
  typedef const __autoreleasing id *const_autoreleasing_id_pointer;

  (void)strong_id_pointer(wip); // expected-error{{functional-style cast from '__weak id *' to 'strong_id_pointer' (aka '__strong id *') casts away qualifiers}}
  (void)weak_id_pointer(sip); // expected-error{{functional-style cast from '__strong id *' to 'weak_id_pointer' (aka '__weak id *') casts away qualifiers}}
  (void)autoreleasing_id_pointer(sip); // expected-error{{functional-style cast from '__strong id *' to 'autoreleasing_id_pointer' (aka '__autoreleasing id *') casts away qualifiers}}
  (void)autoreleasing_id_pointer(wip); // expected-error{{functional-style cast from '__weak id *' to 'autoreleasing_id_pointer' (aka '__autoreleasing id *') casts away qualifiers}}
  (void)const_strong_id_pointer(wip); // expected-error{{functional-style cast from '__weak id *' to 'const_strong_id_pointer' (aka 'const __strong id *') casts away qualifiers}}
  (void)const_weak_id_pointer(sip); // expected-error{{functional-style cast from '__strong id *' to 'const_weak_id_pointer' (aka 'const __weak id *') casts away qualifiers}}
  (void)const_autoreleasing_id_pointer(sip);
  (void)const_autoreleasing_id_pointer(aip);
  (void)const_autoreleasing_id_pointer(wip); // expected-error{{functional-style cast from '__weak id *' to 'const_autoreleasing_id_pointer' (aka 'const __autoreleasing id *') casts away qualifiers}}
}

void test_unsafe_unretained(__strong id *sip, __weak id *wip,
                            __autoreleasing id *aip,
                            __unsafe_unretained id *uip,
                            const __unsafe_unretained id *cuip) {
  uip = sip; // expected-error{{assigning '__strong id *' to '__unsafe_unretained id *' changes retain/release properties of pointer}}
  uip = wip; // expected-error{{assigning '__weak id *' to '__unsafe_unretained id *' changes retain/release properties of pointer}}
  uip = aip; // expected-error{{assigning '__autoreleasing id *' to '__unsafe_unretained id *' changes retain/release properties of pointer}}

  cuip = sip;
  cuip = wip; // expected-error{{assigning '__weak id *' to '__unsafe_unretained id const *' changes retain/release properties of pointer}}
  cuip = aip;
}

void to_void(__strong id *sip, __weak id *wip,
             __autoreleasing id *aip,
             __unsafe_unretained id *uip) {
  void *vp1 = sip;
  void *vp2 = wip;
  void *vp3 = aip;
  void *vp4 = uip;
  (void)(void*)sip;
  (void)(void*)wip;
  (void)(void*)aip;
  (void)(void*)uip;
  (void)static_cast<void*>(sip);
  (void)static_cast<void*>(wip);
  (void)static_cast<void*>(aip);
  (void)static_cast<void*>(uip);
  (void)reinterpret_cast<void*>(sip);
  (void)reinterpret_cast<void*>(wip);
  (void)reinterpret_cast<void*>(aip);
  (void)reinterpret_cast<void*>(uip);

  (void)(void*)&sip;
  (void)(void*)&wip;
  (void)(void*)&aip;
  (void)(void*)&uip;
  (void)static_cast<void*>(&sip);
  (void)static_cast<void*>(&wip);
  (void)static_cast<void*>(&aip);
  (void)static_cast<void*>(&uip);
  (void)reinterpret_cast<void*>(&sip);
  (void)reinterpret_cast<void*>(&wip);
  (void)reinterpret_cast<void*>(&aip);
  (void)reinterpret_cast<void*>(&uip);
}

void from_void(void *vp) {
  __strong id *sip = (__strong id *)vp;
  __weak id *wip = (__weak id *)vp;
  __autoreleasing id *aip = (__autoreleasing id *)vp;
  __unsafe_unretained id *uip = (__unsafe_unretained id *)vp;
  __strong id *sip2 = static_cast<__strong id *>(vp);
  __weak id *wip2 = static_cast<__weak id *>(vp);
  __autoreleasing id *aip2 = static_cast<__autoreleasing id *>(vp);
  __unsafe_unretained id *uip2 = static_cast<__unsafe_unretained id *>(vp);
  __strong id *sip3 = reinterpret_cast<__strong id *>(vp);
  __weak id *wip3 = reinterpret_cast<__weak id *>(vp);
  __autoreleasing id *aip3 = reinterpret_cast<__autoreleasing id *>(vp);
  __unsafe_unretained id *uip3 = reinterpret_cast<__unsafe_unretained id *>(vp);

  __strong id **sipp = (__strong id **)vp;
  __weak id **wipp = (__weak id **)vp;
  __autoreleasing id **aipp = (__autoreleasing id **)vp;
  __unsafe_unretained id **uipp = (__unsafe_unretained id **)vp;

  sip = vp; // expected-error{{assigning to '__strong id *' from incompatible type 'void *'}}
  wip = vp; // expected-error{{assigning to '__weak id *' from incompatible type 'void *'}}
  aip = vp; // expected-error{{assigning to '__autoreleasing id *' from incompatible type 'void *'}}
  uip = vp; // expected-error{{assigning to '__unsafe_unretained id *' from incompatible type 'void *'}}
}

typedef void (^Block)();
typedef void (^Block_strong)() __strong;
typedef void (^Block_autoreleasing)() __autoreleasing;

@class NSString;

void ownership_transfer_in_cast(void *vp, Block *pblk) {
  __strong NSString **sip2 = static_cast<NSString **>(static_cast<__strong id *>(vp));
  __strong NSString **&si2pref = static_cast<NSString **&>(sip2);
  __weak NSString **wip2 = static_cast<NSString **>(static_cast<__weak id *>(vp));
  __autoreleasing id *aip2 = static_cast<id *>(static_cast<__autoreleasing id *>(vp));
  __unsafe_unretained id *uip2 = static_cast<id *>(static_cast<__unsafe_unretained id *>(vp));
  __strong id *sip3 = reinterpret_cast<id *>(reinterpret_cast<__strong id *>(vp));
  __weak id *wip3 = reinterpret_cast<id *>(reinterpret_cast<__weak id *>(vp));
  __autoreleasing id *aip3 = reinterpret_cast<id *>(reinterpret_cast<__autoreleasing id *>(vp));
  __unsafe_unretained id *uip3 = reinterpret_cast<id *>(reinterpret_cast<__unsafe_unretained id *>(vp));

  Block_strong blk_strong1;
  Block_strong blk_strong2 = static_cast<Block>(blk_strong1);
  Block_autoreleasing *blk_auto = static_cast<Block*>(pblk);
}

// Make sure we don't crash.
void writeback_test(NSString & &) {} // expected-error {{type name declared as a reference to a reference}}

void test_strong_opaque() {
  __strong NSString *sptr;
  void *vptr;

  (void)(0 ? sptr : vptr); // expected-error{{operands to conditional of types 'NSString *' and 'void *' are incompatible in ARC mode}}
}
