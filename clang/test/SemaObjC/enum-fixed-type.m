// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#if !__has_feature(objc_fixed_enum)
#  error Enumerations with a fixed underlying type are not supported
#endif

typedef long Integer;

typedef enum : Integer { Enumerator1, Enumerator2 } Enumeration;

int array[sizeof(Enumeration) == sizeof(long)? 1 : -1];


enum Color { Red, Green, Blue };

struct X { 
  enum Color : 4;
  enum Color field1: 4;
  enum Other : Integer field2;
  enum Other : Integer field3 : 4;
  enum  : Integer { Blah, Blarg } field4 : 4;
};

void test() {
  long value = 2;
  Enumeration e = value;
}

// <rdar://10381507>
typedef enum : long { Foo } IntegerEnum;
int arr[(sizeof(typeof(Foo)) == sizeof(typeof(IntegerEnum))) - 1];
int arr1[(sizeof(typeof(Foo)) == sizeof(typeof(long))) - 1];
int arr2[(sizeof(typeof(IntegerEnum)) == sizeof(typeof(long))) - 1];

// <rdar://problem/10760113>
typedef enum : long long { Bar = -1 } LongLongEnum;
int arr3[(long long)Bar == (long long)-1 ? 1 : -1];
