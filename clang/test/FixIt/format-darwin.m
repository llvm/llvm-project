// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -fblocks -Wformat-non-iso -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -fblocks -Wformat-non-iso -verify %s

// RUN: %clang_cc1 -triple i386-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -triple i386-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck -check-prefix=CHECK-32 %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fdiagnostics-parseable-fixits -fblocks -Wformat-non-iso %s 2>&1 | FileCheck -check-prefix=CHECK-64 %s

int printf(const char * restrict, ...);

#if __LP64__
typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef int SInt32;
typedef unsigned int UInt32;

#else

typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef long SInt32;
typedef unsigned long UInt32;
#endif

NSInteger getNSInteger();
NSUInteger getNSUInteger();
SInt32 getSInt32();
UInt32 getUInt32();

void testCorrectionInAllCases() {
  printf("%s", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%s", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%s", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK: fix-it:"{{.*}}":{32:11-32:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{32:16-32:16}:"(long)"

  // CHECK: fix-it:"{{.*}}":{33:11-33:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{33:16-33:16}:"(unsigned long)"

  // CHECK: fix-it:"{{.*}}":{34:11-34:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{34:16-34:16}:"(int)"

  // CHECK: fix-it:"{{.*}}":{35:11-35:13}:"%u"
  // CHECK: fix-it:"{{.*}}":{35:16-35:16}:"(unsigned int)"
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

  // These cases match the cases in CheckPrintfHandler::checkFormatExpr.
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

  // CHECK: fix-it:"{{.*}}":{81:11-81:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{81:16-81:16}:"(long)("
  // CHECK: fix-it:"{{.*}}":{81:25-81:25}:")"
}


#if __LP64__

void testWarn() {
  printf("%d", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%u", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%ld", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%lu", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK-64: fix-it:"{{.*}}":{92:11-92:13}:"%ld"
  // CHECK-64: fix-it:"{{.*}}":{92:16-92:16}:"(long)"

  // CHECK-64: fix-it:"{{.*}}":{93:11-93:13}:"%lu"
  // CHECK-64: fix-it:"{{.*}}":{93:16-93:16}:"(unsigned long)"

  // CHECK-64: fix-it:"{{.*}}":{94:11-94:14}:"%d"
  // CHECK-64: fix-it:"{{.*}}":{94:17-94:17}:"(int)"

  // CHECK-64: fix-it:"{{.*}}":{95:11-95:14}:"%u"
  // CHECK-64: fix-it:"{{.*}}":{95:17-95:17}:"(unsigned int)"
}

void testPreserveHex() {
  printf("%x", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%x", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}

  // CHECK-64: fix-it:"{{.*}}":{111:11-111:13}:"%lx"
  // CHECK-64: fix-it:"{{.*}}":{111:16-111:16}:"(long)"

  // CHECK-64: fix-it:"{{.*}}":{112:11-112:13}:"%lx"
  // CHECK-64: fix-it:"{{.*}}":{112:16-112:16}:"(unsigned long)"
}

void testNoWarn() {
  printf("%ld", getNSInteger()); // no-warning
  printf("%lu", getNSUInteger()); // no-warning
  printf("%d", getSInt32()); // no-warning
  printf("%u", getUInt32()); // no-warning
}

#else

void testWarn() {
  printf("%ld", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%lu", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%d", getSInt32()); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%u", getUInt32()); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK-32: fix-it:"{{.*}}":{131:17-131:17}:"(long)"

  // CHECK-32: fix-it:"{{.*}}":{132:17-132:17}:"(unsigned long)"

  // CHECK-32: fix-it:"{{.*}}":{133:16-133:16}:"(int)"

  // CHECK-32: fix-it:"{{.*}}":{134:16-134:16}:"(unsigned int)"
}

void testPreserveHex() {
  printf("%lx", getNSInteger()); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%lx", getNSUInteger()); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}

  // CHECK-32: fix-it:"{{.*}}":{146:17-146:17}:"(long)"

  // CHECK-32: fix-it:"{{.*}}":{147:17-147:17}:"(unsigned long)"
}

void testNoWarn() {
  printf("%d", getNSInteger()); // no-warning
  printf("%u", getNSUInteger()); // no-warning
  printf("%ld", getSInt32()); // no-warning
  printf("%lu", getUInt32()); // no-warning
}

#endif


void testCasts() {
  printf("%s", (NSInteger)0); // expected-warning{{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  printf("%s", (NSUInteger)0); // expected-warning{{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
  printf("%s", (SInt32)0); // expected-warning{{values of type 'SInt32' should not be used as format arguments; add an explicit cast to 'int' instead}}
  printf("%s", (UInt32)0); // expected-warning{{values of type 'UInt32' should not be used as format arguments; add an explicit cast to 'unsigned int' instead}}

  // CHECK: fix-it:"{{.*}}":{165:11-165:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{165:16-165:27}:"(long)"

  // CHECK: fix-it:"{{.*}}":{166:11-166:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{166:16-166:28}:"(unsigned long)"

  // CHECK: fix-it:"{{.*}}":{167:11-167:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{167:16-167:24}:"(int)"

  // CHECK: fix-it:"{{.*}}":{168:11-168:13}:"%u"
  // CHECK: fix-it:"{{.*}}":{168:16-168:24}:"(unsigned int)"
}

void testCapitals() {
  printf("%D", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'd'?}}
  printf("%U", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'u'?}}
  printf("%O", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'o'?}}
  
  // CHECK: fix-it:"{{.*}}":{184:12-184:13}:"d"
  // CHECK: fix-it:"{{.*}}":{185:12-185:13}:"u"
  // CHECK: fix-it:"{{.*}}":{186:12-186:13}:"o"

  
  printf("%lD", 1); // expected-warning{{conversion specifier is not supported by ISO C}} expected-note {{did you mean to use 'd'?}} expected-warning{{format specifies type 'long' but the argument has type 'int'}}

  // FIXME: offering two somewhat-conflicting fixits is less than ideal.
  // CHECK: fix-it:"{{.*}}":{193:13-193:14}:"d"
  // CHECK: fix-it:"{{.*}}":{193:11-193:14}:"%D"
}
