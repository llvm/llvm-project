// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.coreFoundation.CFNumber,osx.cocoa.RetainCount -analyzer-store=region -verify -triple x86_64-apple-darwin9 %s

typedef signed long CFIndex;
typedef const struct __CFAllocator * CFAllocatorRef;
enum { kCFNumberSInt8Type = 1, kCFNumberSInt16Type = 2,
       kCFNumberSInt32Type = 3, kCFNumberSInt64Type = 4,
       kCFNumberFloat32Type = 5, kCFNumberFloat64Type = 6,
       kCFNumberCharType = 7, kCFNumberShortType = 8,
       kCFNumberIntType = 9, kCFNumberLongType = 10,
       kCFNumberLongLongType = 11, kCFNumberFloatType = 12,
       kCFNumberDoubleType = 13, kCFNumberCFIndexType = 14,
       kCFNumberNSIntegerType = 15, kCFNumberCGFloatType = 16,
       kCFNumberMaxType = 16 };
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
typedef unsigned char Boolean;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
Boolean CFNumberGetValue(CFNumberRef number, CFNumberType theType, void *valuePtr);

__attribute__((cf_returns_retained)) CFNumberRef f1(unsigned char x) {
  return CFNumberCreate(0, kCFNumberSInt16Type, &x);  // expected-warning{{An 8-bit integer is used to initialize a CFNumber object that represents a 16-bit integer; 8 bits of the CFNumber value will be garbage}}
}

__attribute__((cf_returns_retained)) CFNumberRef f2(unsigned short x) {
  return CFNumberCreate(0, kCFNumberSInt8Type, &x); // expected-warning{{A 16-bit integer is used to initialize a CFNumber object that represents an 8-bit integer; 8 bits of the integer value will be lost}}
}

// test that the attribute overrides the naming convention.
__attribute__((cf_returns_not_retained)) CFNumberRef CreateNum(unsigned char x) {
  return CFNumberCreate(0, kCFNumberSInt8Type, &x); // expected-warning{{leak}}
}

__attribute__((cf_returns_retained)) CFNumberRef f3(unsigned i) {
  return CFNumberCreate(0, kCFNumberLongType, &i); // expected-warning{{A 32-bit integer is used to initialize a CFNumber object that represents a 64-bit integer}}
}

unsigned char getValueTest1(CFNumberRef x) {
  unsigned char scalar = 0;
  CFNumberGetValue(x, kCFNumberSInt16Type, &scalar);  // expected-warning{{A CFNumber object that represents a 16-bit integer is used to initialize an 8-bit integer; 8 bits of the CFNumber value will overwrite adjacent storage}}
  return scalar;
}

unsigned char getValueTest2(CFNumberRef x) {
  unsigned short scalar = 0;
  CFNumberGetValue(x, kCFNumberSInt8Type, &scalar);  // expected-warning{{A CFNumber object that represents an 8-bit integer is used to initialize a 16-bit integer; 8 bits of the integer value will be garbage}}
  return scalar;
}
