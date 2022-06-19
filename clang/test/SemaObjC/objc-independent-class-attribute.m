// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://20255473

@interface NSObject @end

typedef NSObject * __attribute__((objc_independent_class))dispatch_queue_t;

typedef struct S {int ii; } * __attribute__((objc_independent_class))dispatch_queue_t_2; // expected-warning {{Objective-C object}}

typedef struct { // expected-warning {{'objc_independent_class' attribute may be put on a typedef only; attribute is ignored}}
   NSObject *__attribute__((objc_independent_class)) ns; // expected-warning {{'objc_independent_class' attribute may be put on a typedef only; attribute is ignored}}
} __attribute__((objc_independent_class)) T;

dispatch_queue_t dispatch_queue_create(void);

@interface DispatchQPointerCastIssue : NSObject {
  NSObject *__attribute__((objc_independent_class)) Ivar; // expected-warning {{'objc_independent_class' attribute may be put on a typedef only; attribute is ignored}}
}

@property (copy) NSObject *__attribute__((objc_independent_class)) Prop; // expected-warning {{'objc_independent_class' attribute may be put on a typedef only; attribute is ignored}}

typedef NSObject * __attribute__((objc_independent_class))dispatch_queue_t_1;

@end

@implementation DispatchQPointerCastIssue
+ (dispatch_queue_t) newDispatchQueue {
    return dispatch_queue_create();
}
@end

NSObject *get_nsobject(void) {
  typedef NSObject * __attribute__((objc_independent_class))dispatch_queue_t;
  dispatch_queue_t dt;
  return dt;
}
