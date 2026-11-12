// RUN: cp %s %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fblocks -Wformat -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s

/* This is a test of code modifications created by darwin format fix-its hints 
   that are provided as part of warning */

int printf(const char * restrict, ...);

#if __LP64__
typedef long CFIndex;
typedef long NSInteger;
typedef unsigned long NSUInteger;
#else
typedef int CFIndex;
typedef int NSInteger;
typedef unsigned int NSUInteger;
#endif
CFIndex getCFIndex(void);
NSInteger getNSInteger(void);
NSUInteger getNSUInteger(void);

#define Log1(...) \
do { \
  printf(__VA_ARGS__); \
} while (0)

#define Log2(...) \
do { \
  printf(__VA_ARGS__); \
  printf(__VA_ARGS__); \
} while (0) \

#define Log3(X, Y, Z) \
do { \
  printf(X, Y); \
  printf(X, Z); \
} while (0) \

void test(void) {
  printf("test 1: %s", getNSInteger()); 
  // CHECK: printf("test 1: %ld", (long)getNSInteger());
  printf("test 2: %s %s", getNSInteger(), getNSInteger());
  // CHECK: printf("test 2: %ld %ld", (long)getNSInteger(), (long)getNSInteger());
  
  Log1("test 3: %s", getNSInteger());
  // CHECK: Log1("test 3: %ld", (long)getNSInteger());
  Log1("test 4: %s %s", getNSInteger(), getNSInteger());
  // CHECK: Log1("test 4: %ld %ld", (long)getNSInteger(), (long)getNSInteger());
  
  Log2("test 5: %s", getNSInteger());
  // CHECK: Log2("test 5: %ld", (long)getNSInteger()); 
  Log2("test 6: %s %s", getNSInteger(), getNSInteger());
  // CHECK: Log2("test 6: %ld %ld", (long)getNSInteger(), (long)getNSInteger());
  
  // Artificial test to check that X (in Log3(X, Y, Z))
  // is modified only according to the diagnostics
  // for the first printf and the modification caused 
  // by the second printf is dropped.
  Log3("test 7: %s", getNSInteger(), getNSUInteger());
  // CHECK: Log3("test 7: %ld", (long)getNSInteger(), (unsigned long)getNSUInteger());
}

#define Outer1(...) \
do { \
  printf(__VA_ARGS__); \
} while (0)

#define Outer2(...) \
do { \
  Outer1(__VA_ARGS__); Outer1(__VA_ARGS__); \
} while (0)

void bug33447(void) {
  Outer2("test 8: %s", getNSInteger());  
  // CHECK: Outer2("test 8: %ld", (long)getNSInteger());
  Outer2("test 9: %s %s", getNSInteger(), getNSInteger());
  // CHECK: Outer2("test 9: %ld %ld", (long)getNSInteger(), (long)getNSInteger());
}

void testCFIndex(void) {
  printf("test 10: %s", getCFIndex()); 
  // CHECK: printf("test 10: %ld", (long)getCFIndex());
  printf("test 11: %s %s", getCFIndex(), getCFIndex());
  // CHECK: printf("test 11: %ld %ld", (long)getCFIndex(), (long)getCFIndex());
}
