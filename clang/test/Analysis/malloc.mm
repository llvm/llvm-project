// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store=region -verify -fblocks %s
#include "Inputs/system-header-simulator-objc.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

// Done with headers. Start testing.
void testNSDatafFreeWhenDoneNoError(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength];
}

void testNSDataFreeWhenDoneYES(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}

void testNSDataFreeWhenDoneYES2(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [[NSData alloc] initWithBytesNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}

void testNSStringFreeWhenDoneYES3(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithBytesNoCopy:data length:dataLength encoding:NSUTF8StringEncoding freeWhenDone:1];
}

void testNSStringFreeWhenDoneYES4(NSUInteger dataLength) {
  unichar *data = (unichar*)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithCharactersNoCopy:data length:dataLength freeWhenDone:1];
  free(data); //expected-warning {{Attempt to free non-owned memory}}
}

void testNSStringFreeWhenDoneYES(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithBytesNoCopy:data length:dataLength encoding:NSUTF8StringEncoding freeWhenDone:1]; // no-warning
}

void testNSStringFreeWhenDoneYES2(NSUInteger dataLength) {
  unichar *data = (unichar*)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithCharactersNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}


void testNSDataFreeWhenDoneNO(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}

void testNSDataFreeWhenDoneNO2(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [[NSData alloc] initWithBytesNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}


void testNSStringFreeWhenDoneNO(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithBytesNoCopy:data length:dataLength encoding:NSUTF8StringEncoding freeWhenDone:0]; // expected-warning{{leak}}
}

void testNSStringFreeWhenDoneNO2(NSUInteger dataLength) {
  unichar *data = (unichar*)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithCharactersNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}

void testRelinquished1() {
  void *data = malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:42 freeWhenDone:1];
  free(data); // expected-warning {{Attempt to free non-owned memory}}
}

void testRelinquished2() {
  void *data = malloc(42);
  NSData *nsdata;
  free(data);
  [NSData dataWithBytesNoCopy:data length:42]; // expected-warning {{Attempt to free released memory}}
}

// Test CF/NS...NoCopy. PR12100: Pointers can escape when custom deallocators are provided.
void testNSDatafFreeWhenDone(NSUInteger dataLength) {
  CFStringRef str;
  char *bytes = (char*)malloc(12);
  str = CFStringCreateWithCStringNoCopy(0, bytes, NSNEXTSTEPStringEncoding, 0); // no warning
  CFRelease(str); // default allocator also frees bytes
}

void stringWithExternalContentsExample(void) {
#define BufferSize 1000
    CFMutableStringRef mutStr;
    UniChar *myBuffer;
 
    myBuffer = (UniChar *)malloc(BufferSize * sizeof(UniChar));
 
    mutStr = CFStringCreateMutableWithExternalCharactersNoCopy(0, myBuffer, 0, BufferSize, kCFAllocatorNull); // expected-warning{{leak}}
 
    CFRelease(mutStr);
    //free(myBuffer);
}

// PR12101 : pointers can escape through custom deallocators set on creation of a container.
void TestCallbackReleasesMemory(CFDictionaryKeyCallBacks keyCallbacks) {
  void *key = malloc(12);
  void *val = malloc(12);
  CFMutableDictionaryRef x = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &keyCallbacks, &kCFTypeDictionaryValueCallBacks);
  CFDictionarySetValue(x, key, val); 
  return;// no-warning
}

NSData *radar10976702() {
  void *bytes = malloc(10);
  return [NSData dataWithBytesNoCopy:bytes length:10]; // no-warning
}

void testBlocks() {
  int *x= (int*)malloc(sizeof(int));
  int (^myBlock)(int) = ^(int num) {
    free(x);
    return num;
  };
  myBlock(3);
}

// Test NSMapInsert. 
@interface NSMapTable : NSObject <NSCopying, NSCoding, NSFastEnumeration>
@end
extern void *NSMapGet(NSMapTable *table, const void *key);
extern void NSMapInsert(NSMapTable *table, const void *key, const void *value);
extern void NSMapInsertKnownAbsent(NSMapTable *table, const void *key, const void *value);
char *strdup(const char *s);

NSString * radar11152419(NSString *string1, NSMapTable *map) {
    const char *strkey = "key";
    NSString *string = ( NSString *)NSMapGet(map, strkey);
    if (!string) {
        string = [string1 copy];
        NSMapInsert(map, strdup(strkey), (void*)string); // no warning
        NSMapInsertKnownAbsent(map, strdup(strkey), (void*)string); // no warning
    }
    return string;
}

// Test that we handle pointer escaping through OSAtomicEnqueue.
typedef volatile struct {
 void *opaque1;
 long opaque2;
} OSQueueHead;
void OSAtomicEnqueue( OSQueueHead *__list, void *__new, size_t __offset) __attribute__((weak_import));
static inline void radar11111210(OSQueueHead *pool) {
    void *newItem = malloc(4);
    OSAtomicEnqueue(pool, newItem, 4);
}

// Pointer might escape through CGDataProviderCreateWithData (radar://11187558).
typedef struct CGDataProvider *CGDataProviderRef;
typedef void (*CGDataProviderReleaseDataCallback)(void *info, const void *data,
    size_t size);
extern CGDataProviderRef CGDataProviderCreateWithData(void *info,
    const void *data, size_t size,
    CGDataProviderReleaseDataCallback releaseData)
    __attribute__((visibility("default")));
void *calloc(size_t, size_t);

static void releaseDataCallback (void *info, const void *data, size_t size) {
#pragma unused (info, size)
  free((void*)data);
}
void testCGDataProviderCreateWithData() { 
  void* b = calloc(8, 8);
  CGDataProviderRef p = CGDataProviderCreateWithData(0, b, 8*8, releaseDataCallback);
}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory. (radar://11160612)
extern CGDataProviderRef UnknownFunWithCallback(void *info,
    const void *data, size_t size,
    CGDataProviderReleaseDataCallback releaseData)
    __attribute__((visibility("default")));
void testUnknownFunWithCallBack() { 
  void* b = calloc(8, 8);
  CGDataProviderRef p = UnknownFunWithCallback(0, b, 8*8, releaseDataCallback);
}

// Test blocks.
void acceptBlockParam(void *, void (^block)(void *), unsigned);
void testCallWithBlockCallback() {
  void *l = malloc(12);
  acceptBlockParam(l, ^(void *i) { free(i); }, sizeof(char *));
}

// Test blocks in system headers.
void testCallWithBlockCallbackInSystem() {
  void *l = malloc(12);
  SystemHeaderFunctionWithBlockParam(l, ^(void *i) { free(i); }, sizeof(char *));
}

// Test escape into NSPointerArray. radar://11691035, PR13140
void foo(NSPointerArray* pointerArray) {
  
  void* p1 = malloc (1024);
  if (p1) {
    [pointerArray addPointer:p1];
  }

  void* p2 = malloc (1024);
  if (p2) {
    [pointerArray insertPointer:p2 atIndex:1];
  }

  void* p3 = malloc (1024);
  if (p3) {
    [pointerArray replacePointerAtIndex:1 withPointer:p3];
  }

  // Freeing the buffer is allowed.
  void* buffer = [pointerArray pointerAtIndex:0];
  free(buffer);
}

void noCrashOnVariableArgumentSelector() {
  NSMutableString *myString = [NSMutableString stringWithString:@"some text"];
  [myString appendFormat:@"some text = %d", 3];
}
