// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -fblocks -analyzer-opt-analyze-nested-blocks -verify -Wno-strict-prototypes %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -fblocks -analyzer-opt-analyze-nested-blocks -verify -x objective-c++ %s

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
dispatch_queue_t
dispatch_queue_create(const char *label, dispatch_queue_attr_t attr);


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
- (id)init;
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

struct Block_layout {
  int flags;
};

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
        logQueue = dispatch_queue_create("com.mycompany.myproduct.asl", 0);
        client = asl_open(((char*)0), "com.mycompany.myproduct", 0);
      });
  } while (0);

  va_list args;
  __builtin_va_start(args, format);

  NSString *str = [[NSString alloc] initWithFormat:format arguments:args];
  dispatch_async(logQueue, ^{ asl_log(client, ((aslmsg)0), 4, "%s", [str UTF8String]); });
  [str release];

  __builtin_va_end(args);
}

// test2 - Test that captured variables that are uninitialized are flagged
// as such.
void test2(void) {
  static int y = 0;
  int x;
  ^{ y = x + 1; }();  // expected-warning{{Variable 'x' is uninitialized when captured by block}}
}

void test2_b(void) {
  static int y = 0;
  __block int x;
  ^{ y = x + 1; }(); // expected-warning {{left operand of '+' is a garbage value}}
}

void test2_c(void) {
  typedef void (^myblock)(void);
  myblock f = ^(void) { f(); }; // expected-warning{{Variable 'f' is uninitialized when captured by block}}
}


void testMessaging(void) {
  // <rdar://problem/12119814>
  [[^(void){} copy] release];
}


@interface rdar12415065 : NSObject
@end

@implementation rdar12415065
- (void)test {
  // At one point this crashed because we created a path note at a
  // PreStmtPurgeDeadSymbols point but only knew how to deal with PostStmt
  // points. <rdar://problem/12687586>

  extern dispatch_queue_t queue;

  if (!queue)
    return;

  // This previously was a false positive with 'x' being flagged as being
  // uninitialized when captured by the exterior block (when it is only
  // captured by the interior block).
  dispatch_async(queue, ^{
    double x = 0.0;
    if (24.0f < x) {
      dispatch_async(queue, ^{ (void)x; });
      [self test];
    }
  });
}
@end

void testReturnVariousSignatures(void) {
  (void)^int(void){
    return 42;
  }();

  (void)^int{
    return 42;
  }();

  (void)^(void){
    return 42;
  }();

  (void)^{
    return 42;
  }();
}

// This test used to cause infinite loop in the region invalidation.
void blockCapturesItselfInTheLoop(int x, int m) {
  void (^assignData)(int) = ^(int x){
    x++;
  };
  while (m < 0) {
    void (^loop)(int);
    loop = ^(int x) {
      assignData(x);
    };
    assignData = loop;
    m++;
  }
  assignData(x);
}

// Blocks that called the function they were contained in that also have
// static locals caused crashes.
// rdar://problem/21698099
void takeNonnullBlock(void (^)(void)) __attribute__((nonnull));
void takeNonnullIntBlock(int (^)(void)) __attribute__((nonnull));

void testCallContainingWithSignature1(void)
{
  takeNonnullBlock(^{
    static const char str[] = "Lost connection to sharingd";
    testCallContainingWithSignature1();
  });
}

void testCallContainingWithSignature2(void)
{
  takeNonnullBlock(^void{
    static const char str[] = "Lost connection to sharingd";
    testCallContainingWithSignature2();
  });
}

void testCallContainingWithSignature3(void)
{
  takeNonnullBlock(^void(void){
    static const char str[] = "Lost connection to sharingd";
    testCallContainingWithSignature3();
  });
}

void testCallContainingWithSignature4(void)
{
  takeNonnullBlock(^void(void){
    static const char str[] = "Lost connection to sharingd";
    testCallContainingWithSignature4();
  });
}

void testCallContainingWithSignature5(void)
{
  takeNonnullIntBlock(^{
    static const char str[] = "Lost connection to sharingd";
    testCallContainingWithSignature5();
    return 0;
  });
}

__attribute__((objc_root_class))
@interface SuperClass
- (void)someMethod;
@end

@interface SomeClass : SuperClass
@end

// Make sure to properly handle super-calls when a block captures
// a local variable named 'self'.
@implementation SomeClass
-(void)foo; {
  /*__weak*/ SomeClass *weakSelf = self;
  (void)(^(void) {
    SomeClass *self = weakSelf;
    (void)(^(void) {
      (void)self;
      [super someMethod]; // no-warning
    });
  });
}
@end

// The incorrect block variable initialization below is a hard compile-time
// error in C++.
#if !defined(__cplusplus)
void call_block_with_fewer_arguments(void) {
  void (^b)() = ^(int a) { };
  b(); // expected-warning {{Block taking 1 argument is called with fewer (0)}}
}
#endif

int getBlockFlags(void) {
  int x = 0;
  return ((struct Block_layout *)^{ (void)x; })->flags; // no-warning
}
