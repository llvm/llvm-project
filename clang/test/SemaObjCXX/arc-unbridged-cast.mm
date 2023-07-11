// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s

typedef const struct __CFString * CFStringRef;
typedef const void * CFTypeRef;
extern "C" CFTypeRef CFBridgingRetain(id X);
extern "C" id CFBridgingRelease(CFTypeRef);


@interface Object
@property CFStringRef property;
- (CFStringRef) implicitProperty;
- (CFStringRef) newString;
- (CFStringRef) makeString;
@end

extern Object *object;

id test0(void) {
  id p1 = (id)[object property];
  id p2 = (__bridge_transfer id)[object property];
  id p3 = (__bridge id)[object property];
  return (id) object.property;
}

CFStringRef unauditedString(void);
CFStringRef plusOneString(void) __attribute__((cf_returns_retained));

#pragma clang arc_cf_code_audited begin
CFStringRef auditedString(void);
CFStringRef auditedCreateString(void);
#pragma clang arc_cf_code_audited end

void test1(int cond) {
  id x;
  x = (id) auditedString();
  x = (id) (cond ? auditedString() : (void*) 0);
  x = (id) (cond ? (void*) 0 : auditedString());
  x = (id) (cond ? (CFStringRef) @"help" : auditedString());

  x = (id) unauditedString(); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? unauditedString() : (void*) 0); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? (void*) 0 : unauditedString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? (CFStringRef) @"help" : unauditedString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRelease call to}}

  x = (id) auditedCreateString(); // expected-error {{requires a bridged cast}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? auditedCreateString() : (void*) 0); // expected-error {{requires a bridged cast}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? (void*) 0 : auditedCreateString()); // expected-error {{requires a bridged cast}} expected-note {{use CFBridgingRelease call to}}
  x = (id) (cond ? (CFStringRef) @"help" : auditedCreateString()); // expected-error {{requires a bridged cast}} expected-note {{use CFBridgingRelease call to}}

  x = (id) [object property];
  x = (id) (cond ? [object property] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object property]);
  x = (id) (cond ? (CFStringRef) @"help" : [object property]);  

  x = (id) object.property;
  x = (id) (cond ? object.property : (void*) 0);
  x = (id) (cond ? (void*) 0 : object.property);
  x = (id) (cond ? (CFStringRef) @"help" : object.property);  

  x = (id) object.implicitProperty;
  x = (id) (cond ? object.implicitProperty : (void*) 0);
  x = (id) (cond ? (void*) 0 : object.implicitProperty);
  x = (id) (cond ? (CFStringRef) @"help" : object.implicitProperty);  

  x = (id) [object makeString];
  x = (id) (cond ? [object makeString] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object makeString]);
  x = (id) (cond ? (CFStringRef) @"help" : [object makeString]);  

  x = (id) [object newString];
  x = (id) (cond ? [object newString] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object newString]);
  x = (id) (cond ? (CFStringRef) @"help" : [object newString]); // a bit questionable
}

@interface CFTaker
- (void) takeOrdinary: (CFStringRef) arg;
- (void) takeVariadic: (int) n, ...;
- (void) takeConsumed: (CFStringRef __attribute__((cf_consumed))) arg;
@end
void testCFTaker(CFTaker *taker, id string) {
  [taker takeOrdinary: (CFStringRef) string];
  [taker takeVariadic: 1, (CFStringRef) string];
  [taker takeConsumed: (CFStringRef) string]; // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}
}

void takeCFOrdinaryUnaudited(CFStringRef arg);
void takeCFVariadicUnaudited(int n, ...);
void takeCFConsumedUnaudited(CFStringRef __attribute__((cf_consumed)) arg);
#pragma clang arc_cf_code_audited begin
void takeCFOrdinaryAudited(CFStringRef arg);
void takeCFVariadicAudited(int n, ...);
void takeCFConsumedAudited(CFStringRef __attribute__((cf_consumed)) arg);
#pragma clang arc_cf_code_audited end

void testTakerFunctions(id string) {
  takeCFOrdinaryUnaudited((CFStringRef) string); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}
  takeCFVariadicUnaudited(1, (CFStringRef) string); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}
  takeCFConsumedUnaudited((CFStringRef) string); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}

  void (*taker)(CFStringRef) = 0;
  taker((CFStringRef) string); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}

  takeCFOrdinaryAudited((CFStringRef) string);
  takeCFVariadicAudited(1, (CFStringRef) string);
  takeCFConsumedAudited((CFStringRef) string); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef'}} expected-note {{use __bridge to}} expected-note {{use CFBridgingRetain call to}}
}

id obj;

void rdar12788838() {
  void *foo = reinterpret_cast<void *>(obj); // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}} \
		// expected-note {{use __bridge with C-style cast to convert directly}} \
		// expected-note {{use CFBridgingRetain call to make an ARC object available as a +1 'void *'}}
}
