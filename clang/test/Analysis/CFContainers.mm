// RUN: %clang_cc1 -analyze -analyzer-checker=osx.coreFoundation.containers.PointerSizedValues,osx.coreFoundation.containers.OutOfBounds -analyzer-store=region -triple x86_64-apple-darwin -verify %s

typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFString * CFStringRef;
typedef unsigned char Boolean;
typedef signed long CFIndex;
extern
const CFAllocatorRef kCFAllocatorDefault;
typedef const void * (*CFArrayRetainCallBack)(CFAllocatorRef allocator, const void *value);
typedef void (*CFArrayReleaseCallBack)(CFAllocatorRef allocator, const void *value);
typedef CFStringRef (*CFArrayCopyDescriptionCallBack)(const void *value);
typedef Boolean (*CFArrayEqualCallBack)(const void *value1, const void *value2);
typedef struct {
    CFIndex version;
    CFArrayRetainCallBack retain;
    CFArrayReleaseCallBack release;
    CFArrayCopyDescriptionCallBack copyDescription;
    CFArrayEqualCallBack equal;
} CFArrayCallBacks;
typedef const struct __CFArray * CFArrayRef;
CFArrayRef CFArrayCreate(CFAllocatorRef allocator, const void **values, CFIndex numValues, const CFArrayCallBacks *callBacks);
typedef const struct __CFString * CFStringRef;
enum {
    kCFNumberSInt8Type = 1,
    kCFNumberSInt16Type = 2,
    kCFNumberSInt32Type = 3,
    kCFNumberSInt64Type = 4,
    kCFNumberFloat32Type = 5,
    kCFNumberFloat64Type = 6,
    kCFNumberCharType = 7,
    kCFNumberShortType = 8,
    kCFNumberIntType = 9,
    kCFNumberLongType = 10,
    kCFNumberLongLongType = 11,
    kCFNumberFloatType = 12,
    kCFNumberDoubleType = 13,
    kCFNumberCFIndexType = 14,
    kCFNumberNSIntegerType = 15,
    kCFNumberCGFloatType = 16,
    kCFNumberMaxType = 16
};
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
typedef CFIndex CFComparisonResult;
typedef const struct __CFDictionary * CFDictionaryRef;
typedef const void * (*CFDictionaryRetainCallBack)(CFAllocatorRef allocator, const void *value);
typedef void (*CFDictionaryReleaseCallBack)(CFAllocatorRef allocator, const void *value);
typedef CFStringRef (*CFDictionaryCopyDescriptionCallBack)(const void *value);
typedef Boolean (*CFDictionaryEqualCallBack)(const void *value1, const void *value2);
typedef Boolean (*CFArrayEqualCallBack)(const void *value1, const void *value2);
typedef Boolean (*CFSetEqualCallBack)(const void *value1, const void *value2);
typedef const void * (*CFSetRetainCallBack)(CFAllocatorRef allocator, const void *value);
typedef void (*CFSetReleaseCallBack)(CFAllocatorRef allocator, const void *value);
typedef CFStringRef (*CFSetCopyDescriptionCallBack)(const void *value);
typedef struct {
    CFIndex version;
    CFSetRetainCallBack retain;
    CFSetReleaseCallBack release;
    CFSetCopyDescriptionCallBack copyDescription;
    CFSetEqualCallBack equal;
} CFSetCallBacks;
typedef struct {
    CFIndex version;
    CFDictionaryRetainCallBack retain;
    CFDictionaryReleaseCallBack release;
    CFDictionaryCopyDescriptionCallBack copyDescription;
    CFDictionaryEqualCallBack equal;
} CFDictionaryKeyCallBacks;
typedef struct {
    CFIndex version;
    CFDictionaryRetainCallBack retain;
    CFDictionaryReleaseCallBack release;
    CFDictionaryCopyDescriptionCallBack copyDescription;
    CFDictionaryEqualCallBack equal;
} CFDictionaryValueCallBacks;
CFDictionaryRef CFDictionaryCreate(CFAllocatorRef allocator, const void **keys, const void **values, CFIndex numValues, const CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
extern
const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
typedef const struct __CFSet * CFSetRef;
extern
const CFSetCallBacks kCFTypeSetCallBacks;
extern
const CFDictionaryKeyCallBacks kCFCopyStringDictionaryKeyCallBacks;
extern
const void *CFArrayGetValueAtIndex(CFArrayRef theArray, CFIndex idx);
extern
CFIndex CFArrayGetCount(CFArrayRef theArray);
CFDictionaryRef CFDictionaryCreate(CFAllocatorRef allocator, const void **keys, const void **values, CFIndex numValues, const 
CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
extern
CFSetRef CFSetCreate(CFAllocatorRef allocator, const void **values, CFIndex numValues, const CFSetCallBacks *callBacks);
#define CFSTR(cStr)  ((CFStringRef) __builtin___CFStringMakeConstantString ("" cStr ""))
#define NULL __null

// Done with the headers. 
// Test alpha.osx.cocoa.ContainerAPI checker.
void testContainers(int **xNoWarn, CFIndex count) {
  int x[] = { 1, 2, 3 };
  CFArrayRef foo = CFArrayCreate(kCFAllocatorDefault, (const void **) x, sizeof(x) / sizeof(x[0]), 0);// expected-warning {{The second argument to 'CFArrayCreate' must be a C array of pointer-sized}}

  CFArrayRef fooNoWarn = CFArrayCreate(kCFAllocatorDefault, (const void **) xNoWarn, sizeof(xNoWarn) / sizeof(xNoWarn[0]), 0); // no warning
  CFArrayRef fooNoWarn2 = CFArrayCreate(kCFAllocatorDefault, 0, sizeof(xNoWarn) / sizeof(xNoWarn[0]), 0);// no warning, passing in 0
  CFArrayRef fooNoWarn3 = CFArrayCreate(kCFAllocatorDefault, NULL, sizeof(xNoWarn) / sizeof(xNoWarn[0]), 0);// no warning, passing in NULL

  CFSetRef set = CFSetCreate(NULL, (const void **)x, 3, &kCFTypeSetCallBacks); // expected-warning {{The second argument to 'CFSetCreate' must be a C array of pointer-sized values}}
  CFArrayRef* pairs = new CFArrayRef[count];
  CFSetRef fSet = CFSetCreate(kCFAllocatorDefault, (const void**) pairs, count - 1, &kCFTypeSetCallBacks);// no warning
}

void CreateDict(int *elems) {
  const short days28 = 28;
  const short days30 = 30;
  const short days31 = 31;
  CFIndex numValues = 6;  
  CFStringRef keys[6];
  CFNumberRef values[6];
  keys[0] = CFSTR("January");  values[0] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days31);
  keys[1] = CFSTR("February"); values[1] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days28);
  keys[2] = CFSTR("March"); values[2] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days31);
  keys[3] = CFSTR("April"); values[3] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days30);
  keys[4] = CFSTR("May"); values[4] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days31);
  keys[5] = CFSTR("June"); values[5] = CFNumberCreate(kCFAllocatorDefault, kCFNumberShortType, &days30);

  const CFDictionaryKeyCallBacks keyCB = kCFCopyStringDictionaryKeyCallBacks;
  const CFDictionaryValueCallBacks valCB = kCFTypeDictionaryValueCallBacks;
  CFDictionaryRef dict1 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)keys, (const void**)values, numValues, &keyCB, &valCB); // no warning
  CFDictionaryRef dict2 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)elems[0], (const void**)values, numValues, &keyCB, &valCB); //expected-warning {{The second argument to 'CFDictionaryCreate' must be a C array of}} expected-warning {{cast to 'const void **' from smaller integer type 'int'}}
  CFDictionaryRef dict3 = CFDictionaryCreate(kCFAllocatorDefault, (const void**)keys, (const void**)elems, numValues, &keyCB, &valCB); // expected-warning {{The third argument to 'CFDictionaryCreate' must be a C array of pointer-sized values}}
}

void OutOfBoundsSymbolicOffByOne(const void ** input, CFIndex S) {
  CFArrayRef array;
  array = CFArrayCreate(kCFAllocatorDefault, input, S, 0);
  const void *s1 = CFArrayGetValueAtIndex(array, 0);   // no warning
  const void *s2 = CFArrayGetValueAtIndex(array, S-1); // no warning
  const void *s3 = CFArrayGetValueAtIndex(array, S);   // expected-warning {{Index is out of bounds}}
}

void OutOfBoundsConst(const void ** input, CFIndex S) {
  CFArrayRef array;
  array = CFArrayCreate(kCFAllocatorDefault, input, 3, 0);
  const void *s1 = CFArrayGetValueAtIndex(array, 0); // no warning
  const void *s2 = CFArrayGetValueAtIndex(array, 2); // no warning
  const void *s3 = CFArrayGetValueAtIndex(array, 5); // expected-warning {{Index is out of bounds}}
  
  // TODO: The solver is probably not strong enough here.
  CFIndex sIndex;
  for (sIndex = 0 ; sIndex <= 5 ; sIndex += 3 ) {
    const void *s = CFArrayGetValueAtIndex(array, sIndex); 
  }  
}

void OutOfBoundsZiro(const void ** input, CFIndex S) {
  CFArrayRef array;
  // The API allows to set the size to 0. Check that we don't undeflow when the size is 0.
  array = CFArrayCreate(kCFAllocatorDefault, 0, 0, 0);
  const void *s1 = CFArrayGetValueAtIndex(array, 0); // expected-warning {{Index is out of bounds}}
}

void TestGetCount(CFArrayRef A, CFIndex sIndex) {
  CFIndex sCount = CFArrayGetCount(A);
  if (sCount > sIndex)
    const void *s1 = CFArrayGetValueAtIndex(A, sIndex);
  const void *s2 = CFArrayGetValueAtIndex(A, sCount);// expected-warning {{Index is out of bounds}}
}

typedef void* XX[3];
void TestPointerToArray(int *elems, void *p1, void *p2, void *p3, unsigned count, void* fn[], char cp[]) {
  void* x[] = { p1, p2, p3 };
  CFArrayCreate(0, (const void **) &x, count, 0); // no warning

  void* y[] = { p1, p2, p3 };
  CFArrayCreate(0, (const void **) y, count, 0); // no warning
  XX *z = &x;
  CFArrayCreate(0, (const void **) z, count, 0); // no warning

  CFArrayCreate(0, (const void **) &fn, count, 0); // false negative
  CFArrayCreate(0, (const void **) fn, count, 0); // no warning
  CFArrayCreate(0, (const void **) cp, count, 0); // expected-warning {{The second argument to 'CFArrayCreate' must be a C array of pointer-sized}}

  char cc[] = { 0, 2, 3 };
  CFArrayCreate(0, (const void **) &cc, count, 0); // expected-warning {{The second argument to 'CFArrayCreate' must be a C array of pointer-sized}}
  CFArrayCreate(0, (const void **) cc, count, 0); // expected-warning {{The second argument to 'CFArrayCreate' must be a C array of pointer-sized}}
}

void TestUndef(CFArrayRef A, CFIndex sIndex, void* x[]) {
  unsigned undefVal;
  const void *s1 = CFArrayGetValueAtIndex(A, undefVal);

  unsigned undefVal2;
  CFArrayRef B = CFArrayCreate(0, (const void **) &x, undefVal2, 0); 
  const void *s2 = CFArrayGetValueAtIndex(B, 2);
}

void TestConst(CFArrayRef A, CFIndex sIndex, void* x[]) {
  CFArrayRef B = CFArrayCreate(0, (const void **) &x, 4, 0); 
  const void *s1 = CFArrayGetValueAtIndex(B, 2);

}

void TestNullArray() {
  CFArrayGetValueAtIndex(0, 0);
}
