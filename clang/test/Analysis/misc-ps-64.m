// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -analyzer-constraints=range -verify -fblocks %s
// expected-no-diagnostics

// <rdar://problem/6440393> - A bunch of misc. failures involving evaluating
//  these expressions and building CFGs.  These tests are here to prevent
//  regressions.
typedef long long int64_t;
@class NSString, NSDictionary;
typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef unsigned char Boolean;
typedef const struct __CFDictionary * CFDictionaryRef;

extern Boolean CFDictionaryGetValueIfPresent(CFDictionaryRef theDict, const void *key, const void **value);
void shazam(NSUInteger i, unsigned char **out);

void rdar_6440393_1(NSDictionary *dict) {
  NSInteger x = 0;
  unsigned char buf[10], *bufptr = buf;
  if (!CFDictionaryGetValueIfPresent(0, dict, (void *)&x))
    return;
  shazam(x, &bufptr);
}

// <rdar://problem/6845148> - In this example we got a signedness
// mismatch between the literal '0' and the value of 'scrooge'.  The
// trick is to have the evaluator convert the literal to an unsigned
// integer when doing a comparison with the pointer.  This happens
// because of the transfer function logic of
// OSAtomicCompareAndSwap64Barrier, which doesn't have special casts
// in place to do this for us.
_Bool OSAtomicCompareAndSwap64Barrier( int64_t __oldValue, int64_t __newValue, volatile int64_t *__theValue );
extern id objc_lookUpClass(const char *name);
void rdar_6845148(id debug_yourself) {
  if (!debug_yourself) {
    const char *wacky = ((void *)0);  
    Class scrooge = wacky ? (Class)objc_lookUpClass(wacky) : ((void *)0);  
    OSAtomicCompareAndSwap64Barrier(0, (int64_t)scrooge, (int64_t*)&debug_yourself);
  }
}
void rdar_6845148_b(id debug_yourself) {
  if (!debug_yourself) {
    const char *wacky = ((void *)0);  
    Class scrooge = wacky ? (Class)objc_lookUpClass(wacky) : ((void *)0);  
    OSAtomicCompareAndSwap64Barrier((int64_t)scrooge, 0, (int64_t*)&debug_yourself);
  }
}
