// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -analyzer-store=region -verify -Wno-objc-root-class -fblocks %s
#include "Inputs/system-header-simulator-objc.h"

@class NSString;
typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

// RDar10579586 - Test use of malloc() with Objective-C string literal as a
// test condition.  Not really a malloc() issue, but this also exercises
// the check that malloc() returns uninitialized memory.
@interface RDar10579586
struct rdar0579586_str {
    char str_c;
};
@end

void rdar10579586(char x);

@implementation RDar10579586
+ (NSString *)foobar
{
    struct rdar0579586_str *buffer = ((void*)0);
    NSString *error = ((void*)0);

    if ((buffer = malloc(sizeof(struct rdar0579586_str))) == ((void*)0))
        error = @"buffer allocation failure";

    if (error != ((void*)0))
        return error;

    rdar10579586(buffer->str_c); // expected-warning {{1st function call argument is an uninitialized value}}
    free(buffer);
    return ((void*)0);
}
@end

@interface MyArray : NSObject {
  id * objects;
}
@end

void _ArrayCreate(void) {
  MyArray *array = (MyArray *)malloc(12);
  array = [array init];
  free(array); // no-warning
}

void testNSDataTruePositiveLeak(void) {
  char *b = (char *)malloc(12);
  NSData *d = [[NSData alloc] initWithBytes: b length: 12]; // expected-warning {{Potential leak of memory pointed to by 'b'}}
}

id wrapInNSValue(void) {
  void *buffer = malloc(4);
  return [NSValue valueWithPointer:buffer]; // no-warning
}