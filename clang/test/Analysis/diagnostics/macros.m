// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx -fblocks -analyzer-output=text -verify %s

#include "../Inputs/system-header-simulator-objc.h"

@interface NSDictionary : NSObject
- (NSUInteger)count;
- (id)objectForKey:(id)aKey;
- (NSEnumerator *)keyEnumerator;
@end
@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)anObject forKey:(id <NSCopying>)aKey;
@end

void testBOOLMacro(BOOL b) {
  if (b == YES) { // expected-note {{Assuming 'b' is equal to YES}}
                  // expected-note@-1 {{Taking true branch}}
    char *p = NULL;// expected-note {{'p' initialized to a null pointer value}}
    *p = 7;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}

void testNilMacro(NSMutableDictionary *d, NSObject *o) {
  if (o == nil) // expected-note {{Assuming 'o' is equal to nil}}
                // expected-note@-1 {{Taking true branch}}
    [d setObject:o forKey:[o description]]; // expected-warning {{Key argument to 'setObject:forKey:' cannot be nil}}
                                            // expected-note@-1 {{'description' not called because the receiver is nil}}
                                            // expected-note@-2 {{Key argument to 'setObject:forKey:' cannot be nil}}

  return;
}
