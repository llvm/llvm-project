// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core -analyzer-store=region -fblocks -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from Mac OS X headers:
//===----------------------------------------------------------------------===//

typedef __builtin_va_list va_list;
typedef unsigned int uint32_t;
typedef struct dispatch_queue_s *dispatch_queue_t;
typedef struct dispatch_queue_attr_s *dispatch_queue_attr_t;
typedef void (^dispatch_block_t)(void);
void dispatch_async(dispatch_queue_t queue, dispatch_block_t block);
__attribute__((visibility("default"))) __attribute__((__malloc__)) __attribute__((__warn_unused_result__)) __attribute__((__nothrow__)) dispatch_queue_t dispatch_queue_create(const char *label, dispatch_queue_attr_t attr);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);
typedef signed char BOOL;
typedef unsigned long NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
@end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {}
+ (id)alloc;
- (id)copy;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
- (const char *)UTF8String;
- (id)initWithFormat:(NSString *)format arguments:(va_list)argList __attribute__((format(__NSString__, 1, 0)));
@end
@class NSString, NSData;
typedef struct cssm_sample {} CSSM_SAMPLEGROUP, *CSSM_SAMPLEGROUP_PTR;
typedef struct __aslclient *aslclient;
typedef struct __aslmsg *aslmsg;
aslclient asl_open(const char *ident, const char *facility, uint32_t opts);
int asl_log(aslclient asl, aslmsg msg, int level, const char *format, ...) __attribute__((__format__ (__printf__, 4, 5)));

//===----------------------------------------------------------------------===//
// Begin actual test cases.
//===----------------------------------------------------------------------===//

// test1 - This test case exposed logic that caused the analyzer to crash because of a memory bug
//  in BlockDataRegion.  It represents real code that contains two block literals.  Eventually
//  via IPA 'logQueue' and 'client' should be updated after the call to 'dispatch_once'.
void test1(NSString *format, ...) {
  static dispatch_queue_t logQueue;
  static aslclient client;
  static dispatch_once_t pred;
  do {
    if (__builtin_expect(*(&pred), ~0l) != ~0l)
      dispatch_once(&pred, ^{
        logQueue = dispatch_queue_create("com.mycompany.myproduct.asl", ((void*)0));
        client = asl_open(((void*)0), "com.mycompany.myproduct", 0);
      });
  } while (0);

  va_list args;
  __builtin_va_start(args, format);

  NSString *str = [[NSString alloc] initWithFormat:format arguments:args];
  dispatch_async(logQueue, ^{ asl_log(client, ((void*)0), 4, "%s", [str UTF8String]); });
  [str release];

  __builtin_va_end(args);
}

// test2 - Test that captured variables that are uninitialized are flagged
// as such.
void test2() {
  static int y = 0;
  int x;
  ^{ y = x + 1; }();  // expected-warning{{Variable 'x' is uninitialized when captured by block}}
}

void test2_b() {
  static int y = 0;
  __block int x;
  ^{ y = x + 1; }(); // expected-warning {{left operand of '+' is a garbage value}}
}

void test2_c() {
  typedef void (^myblock)(void);
  myblock f = ^() { f(); }; // expected-warning{{Variable 'f' is uninitialized when captured by block}}
}


void testMessaging() {
  // <rdar://problem/12119814>
  [[^(){} copy] release];
}
