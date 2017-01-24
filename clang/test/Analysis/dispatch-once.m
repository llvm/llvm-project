// RUN: %clang_cc1 -w -fblocks -analyze -analyzer-checker=core,osx.API,unix.Malloc -verify %s
// RUN: %clang_cc1 -w -fblocks -fobjc-arc -analyze -analyzer-checker=core,osx.API,unix.Malloc -verify %s

#include "Inputs/system-header-simulator-objc.h"

typedef unsigned long size_t;
void *calloc(size_t nmemb, size_t size);

typedef void (^dispatch_block_t)(void);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);

void test_stack() {
  dispatch_once_t once;
  dispatch_once(&once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the local variable 'once' for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as 'static'?}}
}

void test_static_local() {
  static dispatch_once_t once;
  dispatch_once(&once, ^{}); // no-warning
}

void test_heap_var() {
  dispatch_once_t *once = calloc(1, sizeof(dispatch_once_t));
  // Use regexps to check that we're NOT suggesting to make this static.
  dispatch_once(once, ^{}); // expected-warning-re{{{{^Call to 'dispatch_once' uses heap-allocated memory for the predicate value.  Using such transient memory for the predicate is potentially dangerous$}}}}
}

void test_external_pointer(dispatch_once_t *once) {
  // External pointer does not necessarily point to the heap.
  dispatch_once(once, ^{}); // no-warning
}

typedef struct {
  dispatch_once_t once;
} Struct;

void test_local_struct() {
  Struct s;
  dispatch_once(&s.once, ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the local variable 's' for the predicate value.}}
}

void test_heap_struct() {
  Struct *s = calloc(1, sizeof(Struct));
  dispatch_once(&s->once, ^{}); // expected-warning{{Call to 'dispatch_once' uses heap-allocated memory for the predicate value.}}
}

@interface Object : NSObject {
@public
  dispatch_once_t once;
  Struct s;
  dispatch_once_t once_array[2];
}
- (void)test_ivar_from_inside;
- (void)test_ivar_struct_from_inside;
@end

@implementation Object
- (void)test_ivar_from_inside {
  dispatch_once(&once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the instance variable 'once' for the predicate value.}}
}
- (void)test_ivar_struct_from_inside {
  dispatch_once(&s.once, ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 's' for the predicate value.}}
}
- (void)test_ivar_array_from_inside {
  dispatch_once(&once_array[1], ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 'once_array' for the predicate value.}}
}
@end

void test_ivar_from_alloc_init() {
  Object *o = [[Object alloc] init];
  dispatch_once(&o->once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the instance variable 'once' for the predicate value.}}
}
void test_ivar_struct_from_alloc_init() {
  Object *o = [[Object alloc] init];
  dispatch_once(&o->s.once, ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 's' for the predicate value.}}
}
void test_ivar_array_from_alloc_init() {
  Object *o = [[Object alloc] init];
  dispatch_once(&o->once_array[1], ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 'once_array' for the predicate value.}}
}

void test_ivar_from_external_obj(Object *o) {
  // ObjC object pointer always points to the heap.
  dispatch_once(&o->once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the instance variable 'once' for the predicate value.}}
}
void test_ivar_struct_from_external_obj(Object *o) {
  dispatch_once(&o->s.once, ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 's' for the predicate value.}}
}
void test_ivar_array_from_external_obj(Object *o) {
  dispatch_once(&o->once_array[1], ^{}); // expected-warning{{Call to 'dispatch_once' uses memory within the instance variable 'once_array' for the predicate value.}}
}

void test_block_var_from_block() {
  __block dispatch_once_t once;
  ^{
    dispatch_once(&once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the block variable 'once' for the predicate value.}}
  };
}

void use_block_var(dispatch_once_t *once);

void test_block_var_from_outside_block() {
  __block dispatch_once_t once;
  ^{
    use_block_var(&once);
  };
  dispatch_once(&once, ^{}); // expected-warning{{Call to 'dispatch_once' uses the block variable 'once' for the predicate value.}}
}

void test_static_var_from_outside_block() {
  static dispatch_once_t once;
  ^{
    dispatch_once(&once, ^{}); // no-warning
  };
}
