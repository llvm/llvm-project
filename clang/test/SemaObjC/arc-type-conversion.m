// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -verify -fblocks %s

typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);

void * cvt(id arg)
{
  void* voidp_val;
  (void)(int*)arg; // expected-error {{cast of an Objective-C pointer to 'int *' is disallowed with ARC}}
  (void)(id)arg;
  (void)(__autoreleasing id*)arg; // expected-error {{cast of an Objective-C pointer to '__autoreleasing id *' is disallowed with ARC}}
  (void)(id*)arg; // expected-error {{cast of an Objective-C pointer to '__strong id *' is disallowed with ARC}}

  (void)(__autoreleasing id**)voidp_val;
  (void)(void*)voidp_val;
  (void)(void**)arg; // expected-error {{cast of an Objective-C pointer to 'void **' is disallowed with ARC}}
  cvt((void*)arg); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}} \
                   // expected-error {{implicit conversion of C pointer type 'void *' to Objective-C pointer type 'id' requires a bridged cast}} \
                   // expected-note 2 {{use __bridge to convert directly (no change in ownership)}} \
                   // expected-note {{use CFBridgingRetain call to make an ARC object available as a +1 'void *'}} \
                   // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'void *' into ARC}}
  cvt(0);
  (void)(__strong id**)(0);
  return arg; // expected-error {{implicit conversion of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}} \
                   // expected-note {{use __bridge to convert directly (no change in ownership)}} \
                   // expected-note {{use CFBridgingRetain call to make an ARC object available as a +1 'void *'}}
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
  (void)(void*)&sip;
  (void)(void*)&wip;
  (void)(void*)&aip;
  (void)(void*)&uip;
}

void from_void(void *vp) {
  __strong id *sip = (__strong id *)vp;
  __weak id *wip = (__weak id *)vp;
  __autoreleasing id *aip = (__autoreleasing id *)vp;
  __unsafe_unretained id *uip = (__unsafe_unretained id *)vp;

  __strong id **sipp = (__strong id **)vp;
  __weak id **wipp = (__weak id **)vp;
  __autoreleasing id **aipp = (__autoreleasing id **)vp;
  __unsafe_unretained id **uipp = (__unsafe_unretained id **)vp;

  sip = vp; // expected-error{{implicit conversion of a non-Objective-C pointer type 'void *' to '__strong id *' is disallowed with ARC}}
  wip = vp; // expected-error{{implicit conversion of a non-Objective-C pointer type 'void *' to '__weak id *' is disallowed with ARC}}
  aip = vp; // expected-error{{implicit conversion of a non-Objective-C pointer type 'void *' to '__autoreleasing id *' is disallowed with ARC}}
  uip = vp; // expected-error{{implicit conversion of a non-Objective-C pointer type 'void *' to '__unsafe_unretained id *' is disallowed with ARC}}
}

typedef void (^Block)();
typedef void (^Block_strong)() __strong;
typedef void (^Block_autoreleasing)() __autoreleasing;

@class NSString;

void ownership_transfer_in_cast(void *vp, Block *pblk) {
  __strong NSString **sip = (NSString**)(__strong id *)vp;
  __weak NSString **wip = (NSString**)(__weak id *)vp;
  __autoreleasing id *aip = (id*)(__autoreleasing id *)vp;
  __unsafe_unretained id *uip = (id*)(__unsafe_unretained id *)vp;

  __strong id **sipp = (id**)(__strong id **)vp;
  __weak id **wipp = (id**)(__weak id **)vp;
  __autoreleasing id **aipp = (id**)(__autoreleasing id **)vp;
  __unsafe_unretained id **uipp = (id**)(__unsafe_unretained id **)vp;

  Block_strong blk_strong1;
  Block_strong blk_strong2 = (Block)blk_strong1;
  Block_autoreleasing *blk_auto = (Block*)pblk;

  id lv;
  (void)(id)&lv; // expected-error {{cast of an indirect pointer to an Objective-C pointer to 'id'}}
  (void)(id*)lv; // expected-error {{cast of an Objective-C pointer to '__strong id *'}}
  (void)(NSString*)&lv; // expected-error {{cast of an indirect pointer to an Objective-C pointer to 'NSString *'}}
  (void)(NSString**)lv; // expected-error {{cast of an Objective-C pointer to 'NSString *__strong *'}}
  (void)(Block)&lv; // expected-error {{cast of an indirect pointer to an Objective-C pointer to 'Block'}}
  (void)(Block*)lv; // expected-error {{cast of an Objective-C pointer to '__strong Block *'}}
}

// <rdar://problem/10486347>
void conversion_in_conditional(id a, void* b) {
  id c = 1 ? a : b; // expected-error {{operands to conditional of types 'id' and 'void *' are incompatible in ARC mode}}
  id d = 1 ? b : a; // expected-error {{operands to conditional of types 'void *' and 'id' are incompatible in ARC mode}}
}
