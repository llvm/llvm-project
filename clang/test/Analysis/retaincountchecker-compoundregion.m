// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx -fblocks -verify -Wno-objc-root-class %s
typedef const void *CFTypeRef;
enum { kCFNumberSInt8Type = 1, kCFNumberSInt16Type = 2,
       kCFNumberSInt32Type = 3, kCFNumberSInt64Type = 4,
       kCFNumberFloat32Type = 5, kCFNumberFloat64Type = 6,
       kCFNumberCharType = 7, kCFNumberShortType = 8,
       kCFNumberIntType = 9, kCFNumberLongType = 10,
       kCFNumberLongLongType = 11, kCFNumberFloatType = 12,
       kCFNumberDoubleType = 13, kCFNumberCFIndexType = 14,
       kCFNumberNSIntegerType = 15, kCFNumberCGFloatType = 16,
       kCFNumberMaxType = 16 };
typedef const struct __CFAllocator * CFAllocatorRef;
typedef signed long CFIndex;
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);

void foo(CFAllocatorRef allocator) {
  int width = 0;
  int height = 0;
  CFTypeRef* values = (CFTypeRef[]){
    CFNumberCreate(allocator, kCFNumberSInt32Type, &width), //expected-warning{{Potential leak of an object of type 'CFNumberRef'}}
    CFNumberCreate(allocator, kCFNumberSInt32Type, &height), //expected-warning{{Potential leak of an object of type 'CFNumberRef'}}
  };
}
