// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fdiagnostics-parseable-fixits -fblocks %s 2>&1 | FileCheck %s

@class NSString;
extern void NSLog(NSString *, ...);
int printf(const char * restrict, ...) ;

void test_integer_correction (int x) {
  printf("%d", x); // no-warning
  printf("%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  printf("%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{10:11-10:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{11:11-11:14}:"%d"

  NSLog(@"%d", x); // no-warning
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{16:11-16:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{17:11-17:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{18:11-18:13}:"%d"
}

void test_string_correction (char *x) {
  printf("%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'char *'}}
  printf("%s", x); // no-warning
  printf("%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'char *'}}
  // CHECK: fix-it:"{{.*}}":{25:11-25:13}:"%s"
  // CHECK: fix-it:"{{.*}}":{27:11-27:14}:"%s"

  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'char *'}}
  NSLog(@"%s", x); // no-warning
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'char *'}}
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has type 'char *'}}
  // CHECK: fix-it:"{{.*}}":{31:11-31:13}:"%s"
  // CHECK: fix-it:"{{.*}}":{33:11-33:14}:"%s"
  // CHECK: fix-it:"{{.*}}":{34:11-34:13}:"%s"
}

void test_object_correction (id x) {  
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'id'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'id'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'id'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{41:11-41:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{42:11-42:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{43:11-43:14}:"%@"
}

typedef const struct __CFString * __attribute__((NSObject)) CFStringRef;
void test_cf_object_correction (CFStringRef x) {
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'CFStringRef'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'CFStringRef'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'CFStringRef'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{52:11-52:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{53:11-53:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{54:11-54:14}:"%@"
}

typedef void (^block_t)(void);
void test_block_correction (block_t x) {
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'block_t'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'block_t'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'block_t'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{63:11-63:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{64:11-64:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{65:11-65:14}:"%@"
}

void test_class_correction (Class x) {
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'Class'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'Class'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'Class'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{73:11-73:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{74:11-74:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{75:11-75:14}:"%@"
}


typedef enum : int { NSUTF8StringEncoding = 8 } NSStringEncoding;
void test_fixed_enum_correction(NSStringEncoding x) {
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has underlying type 'int'}}
  // CHECK: fix-it:"{{.*}}":{85:11-85:13}:"%d"
}

typedef __SIZE_TYPE__ size_t;
enum SomeSize : size_t { IntegerSize = sizeof(int) };
void test_named_fixed_enum_correction(enum SomeSize x) {
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has underlying type 'size_t' (aka}}
  // CHECK: fix-it:"{{.*}}":{92:11-92:13}:"%zu"
}


typedef unsigned char uint8_t;
void test_char(char c, signed char s, unsigned char u, uint8_t n) {
  NSLog(@"%s", c); // expected-warning{{format specifies type 'char *' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%lf", c); // expected-warning{{format specifies type 'double' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%c"

  NSLog(@"%@", c); // expected-warning{{format specifies type 'id' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%c", c); // no-warning
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  
  NSLog(@"%s", s); // expected-warning{{format specifies type 'char *' but the argument has type 'signed char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%lf", s); // expected-warning{{format specifies type 'double' but the argument has type 'signed char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%c"

  NSLog(@"%@", s); // expected-warning{{format specifies type 'id' but the argument has type 'signed char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%c", s); // no-warning
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"


  NSLog(@"%s", u); // expected-warning{{format specifies type 'char *' but the argument has type 'unsigned char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%lf", u); // expected-warning{{format specifies type 'double' but the argument has type 'unsigned char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%c"

  NSLog(@"%@", u); // expected-warning{{format specifies type 'id' but the argument has type 'unsigned char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%c", u); // no-warning
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"


  NSLog(@"%s", n); // expected-warning{{format specifies type 'char *' but the argument has type 'uint8_t' (aka 'unsigned char')}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%hhu"

  NSLog(@"%lf", n); // expected-warning{{format specifies type 'double' but the argument has type 'uint8_t' (aka 'unsigned char')}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%hhu"

  NSLog(@"%@", n); // expected-warning{{format specifies type 'id' but the argument has type 'uint8_t' (aka 'unsigned char')}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%hhu"

  NSLog(@"%c", n); // no-warning
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%hhu"


  NSLog(@"%s", 'a'); // expected-warning{{format specifies type 'char *' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%lf", 'a'); // expected-warning{{format specifies type 'double' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%c"

  NSLog(@"%@", 'a'); // expected-warning{{format specifies type 'id' but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"

  NSLog(@"%c", 'a'); // no-warning
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"


  NSLog(@"%s", 'abcd'); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"

  NSLog(@"%lf", 'abcd'); // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%d"

  NSLog(@"%@", 'abcd'); // expected-warning{{format specifies type 'id' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
}

void multichar_constants_false_negative() {
  // The value of a multi-character constant is implementation-defined, but
  // almost certainly shouldn't be printed with %c. However, the current
  // type-checker expects %c to correspond to an integer argument, because
  // many C library functions like fgetc() actually return an int (using -1
  // as a sentinel).
  NSLog(@"%c", 'abcd'); // missing-warning{{format specifies type 'char' but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
}


void test_percent_C() {
  const unsigned short data = 'a';
  NSLog(@"%C", data);  // no-warning

  NSLog(@"%C", 0x260300);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unsigned short)"

  typedef unsigned short unichar;
  
  NSLog(@"%C", 0x260300);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unichar)"
  
  NSLog(@"%C", data ? 0x2F0000 : 0x260300); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unichar)("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:42-[[@LINE-3]]:42}:")"

  NSLog(@"%C", 0.0); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'double'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%f"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unichar)"

  NSLog(@"%C", (char)0x260300); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:22}:"(unichar)"

  NSLog(@"%C", 'a'); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'char'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%c"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:22}:"(unichar)"
}


void testSignedness(long i, unsigned long u) {
  printf("%d", u); // expected-warning{{format specifies type 'int' but the argument has type 'unsigned long'}}
  printf("%i", u); // expected-warning{{format specifies type 'int' but the argument has type 'unsigned long'}}
  printf("%u", i); // expected-warning{{format specifies type 'unsigned int' but the argument has type 'long'}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:13}:"%ld"

  printf("%+d", u); // expected-warning{{format specifies type 'int' but the argument has type 'unsigned long'}}

  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:14}:"%+ld"
}

void testSizeTypes() {
  printf("%zu", 0.f); // expected-warning-re{{format specifies type 'size_t' (aka '{{.+}}') but the argument has type 'float'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%f"

  printf("%zd", 0.f); // expected-warning-re{{format specifies type 'ssize_t' (aka '{{.+}}') but the argument has type 'float'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%f"
  
  short x;
  printf("%zn", &x); // expected-warning-re{{format specifies type 'ssize_t *' (aka '{{.+}}') but the argument has type 'short *'}}
  // PrintfSpecifier::fixType doesn't handle %n, so a fix-it is not emitted, 
  // see the comment in PrintfSpecifier::fixType in PrintfFormatString.cpp.
}

typedef __PTRDIFF_TYPE__ ptrdiff_t;
#define __UNSIGNED_PTRDIFF_TYPE__                                              \
  __typeof__(_Generic((__PTRDIFF_TYPE__)0,                                     \
                      long long int : (unsigned long long int)0,               \
                      long int : (unsigned long int)0,                         \
                      int : (unsigned int)0,                                   \
                      short : (unsigned short)0,                               \
                      signed char : (unsigned char)0))

void testPtrDiffTypes() {
  __UNSIGNED_PTRDIFF_TYPE__ p1 = 0;
  printf("%tu", p1);  // No warning.

  printf("%tu", 0.f); // expected-warning-re{{format specifies type 'unsigned ptrdiff_t' (aka '{{.+}}') but the argument has type 'float'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%f"
  
  ptrdiff_t p2 = 0;
  printf("%td", p2);  // No warning.

  printf("%td", 0.f); // expected-warning-re{{format specifies type 'ptrdiff_t' (aka '{{.+}}') but the argument has type 'float'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%f"

  ptrdiff_t p3 = 0;
  printf("%tn", &p3); // No warning.

  short x;
  printf("%tn", &x); // expected-warning-re{{format specifies type 'ptrdiff_t *' (aka '{{.+}}') but the argument has type 'short *'}}
  // PrintfSpecifier::fixType doesn't handle %n, so a fix-it is not emitted,
  // see the comment in PrintfSpecifier::fixType in PrintfFormatString.cpp.
}

void testEnum() {
  typedef enum {
    ImplicitA = 1,
    ImplicitB = 2
  } Implicit;

  typedef enum {
    ImplicitLLA = 0,
    ImplicitLLB = ~0ULL
  } ImplicitLongLong;

  typedef enum : short {
    ExplicitA = 0,
    ExplicitB
  } ExplicitShort;

  printf("%f", (Implicit)0); // expected-warning{{format specifies type 'double' but the argument has underlying type}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%{{[du]}}"

  printf("%f", (ImplicitLongLong)0); // expected-warning{{format specifies type 'double' but the argument has underlying type}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%{{l*[du]}}"

  printf("%f", (ExplicitShort)0); // expected-warning{{format specifies type 'double' but the argument has underlying type 'short'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%hd"
}
