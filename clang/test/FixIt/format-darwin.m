// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -fblocks -Wformat-non-iso -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -fblocks -Wformat-non-iso -verify %s

// RUN: %clang_cc1 -triple i386-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=CHECK-32 %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=CHECK-64 %s

int printf(const char * restrict, ...);

#if __LP64__
typedef long CFIndex;
typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef int SInt32;
typedef unsigned int UInt32;

#else
typedef int CFIndex;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef long SInt32;
typedef unsigned long UInt32;
#endif

typedef SInt32 OSStatus;

typedef enum NSIntegerEnum : NSInteger {
  EnumValueA,
  EnumValueB
} NSIntegerEnum;

CFIndex getCFIndex(void);
NSInteger getNSInteger(void);
NSUInteger getNSUInteger(void);
SInt32 getSInt32(void);
UInt32 getUInt32(void);
NSIntegerEnum getNSIntegerEnum(void);

void testCorrectionInAllCases(void) {
  printf("%s", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%s", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%s", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:11-[[@LINE-5]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:"(long)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:11-[[@LINE-7]]:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:16-[[@LINE-8]]:16}:"(unsigned long)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:11-[[@LINE-9]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:16-[[@LINE-10]]:16}:"(int)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-11]]:11-[[@LINE-11]]:13}:"%u"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:16-[[@LINE-12]]:16}:"(unsigned int)"

  printf("%s", getNSIntegerEnum()); // expected-warning{{enum values with underlying type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:16}:"(long)"

  printf("%s", getCFIndex()); // expected-warning{{values of type 'CFIndex' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:16}:"(long)"
}

@interface Foo {
@public
  NSInteger _value;
}
- (NSInteger)getInteger;

@property NSInteger value;
@end

struct Bar {
  NSInteger value;
};


void testParens(Foo *obj, struct Bar *record) {
  NSInteger arr[4] = {0};
  NSInteger i = 0;

  // These cases match the relevant cases in CheckPrintfHandler::checkFormatExpr.
  printf("%s", arr[0]);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", getNSInteger());  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", i);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", obj->_value);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", [obj getInteger]);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", obj.value);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", record->value);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", (i ? i : i));  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", *arr);  // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-NOT: fix-it:{{.*}}:")"

  printf("%s", i ? i : i); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:16}:"(long)("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:25-[[@LINE-4]]:25}:")"
}


#if __LP64__

void testWarn(void) {
  printf("%d", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%u", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%ld", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%lu", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-5]]:11-[[@LINE-5]]:13}:"%ld"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:"(long)"

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-7]]:11-[[@LINE-7]]:13}:"%lu"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-8]]:16-[[@LINE-8]]:16}:"(unsigned long)"

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-9]]:11-[[@LINE-9]]:14}:"%d"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-10]]:17-[[@LINE-10]]:17}:"(int)"

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-11]]:11-[[@LINE-11]]:14}:"%u"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-12]]:17-[[@LINE-12]]:17}:"(unsigned int)"

  printf("%d", getNSIntegerEnum()); // expected-warning{{enum values with underlying type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:16}:"(long)"

  printf("%d", getCFIndex()); // expected-warning{{values of type 'CFIndex' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:16}:"(long)"
}

void testPreserveHex(void) {
  printf("%x", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%x", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-3]]:11-[[@LINE-3]]:13}:"%lx"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:"(long)"

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-5]]:11-[[@LINE-5]]:13}:"%lx"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:"(unsigned long)"
}

void testSignedness(NSInteger i, NSUInteger u) {
  printf("%d", u); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%i", u); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%u", i); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%lu"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%lu"
  // CHECK-64: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%ld"
}

void testNoWarn(void) {
  printf("%ld", getNSInteger()); // no-warning
  printf("%lu", getNSUInteger()); // no-warning
  printf("%d", getSInt32()); // no-warning
  printf("%u", getUInt32()); // no-warning
  printf("%ld", getNSIntegerEnum()); // no-warning
}

#else

void testWarn(void) {
  printf("%ld", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%lu", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%d", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%u", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-5]]:17-[[@LINE-5]]:17}:"(long)"
  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-5]]:17-[[@LINE-5]]:17}:"(unsigned long)"
  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-5]]:16-[[@LINE-5]]:16}:"(int)"
  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-5]]:16-[[@LINE-5]]:16}:"(unsigned int)"

  printf("%ld", getNSIntegerEnum()); // expected-warning{{enum values with underlying type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"(long)"

  printf("%ld", getCFIndex()); // expected-warning{{values of type 'CFIndex' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"(long)"
}

void testPreserveHex(void) {
  printf("%lx", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%lx", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}

  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:"(long)"
  // CHECK-32: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:"(unsigned long)"
}

void testNoWarn(void) {
  printf("%d", getNSInteger()); // no-warning
  printf("%u", getNSUInteger()); // no-warning
  printf("%ld", getSInt32()); // no-warning
  printf("%lu", getUInt32()); // no-warning
  printf("%d", getNSIntegerEnum()); // no-warning
}

void testSignedness(NSInteger i, NSUInteger u) {
  // It is valid to use a specifier with the opposite signedness as long as
  // the type is correct.
  printf("%d", u); // no-warning
  printf("%i", u); // no-warning
  printf("%u", i); // no-warning
}

#endif


void testCasts(void) {
  printf("%s", (NSInteger)0); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", (NSUInteger)0); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%s", (SInt32)0); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%s", (UInt32)0); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:11-[[@LINE-5]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:27}:"(long)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:11-[[@LINE-7]]:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:16-[[@LINE-8]]:28}:"(unsigned long)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:11-[[@LINE-9]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:16-[[@LINE-10]]:24}:"(int)"

  // CHECK: fix-it:"{{.*}}":{[[@LINE-11]]:11-[[@LINE-11]]:13}:"%u"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:16-[[@LINE-12]]:24}:"(unsigned int)"

  printf("%s", (NSIntegerEnum)0); // expected-warning{{enum values with underlying type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:31}:"(long)"

  printf("%s", (CFIndex)0); // expected-warning{{values of type 'CFIndex' should not be used as format arguments; add an explicit cast to 'long' instead}}
  
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:16-[[@LINE-3]]:25}:"(long)"
}

void testCapitals(void) {
  printf("%D", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'd'?}}
  printf("%U", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'u'?}}
  printf("%O", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'o'?}}
  
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:12-[[@LINE-4]]:13}:"d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:12-[[@LINE-4]]:13}:"u"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:12-[[@LINE-4]]:13}:"o"

  
  printf("%lD", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'd'?}} expected-warning{{format specifies type 'long' but the argument has type 'int'}}

  // FIXME: offering two somewhat-conflicting fixits is less than ideal.
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:14}:"d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:14}:"%D"
}

void testLayeredTypedefs(OSStatus i) {
  printf("%s", i); // expected-warning {{values of type 'OSStatus' should not be used as format arguments}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
}

