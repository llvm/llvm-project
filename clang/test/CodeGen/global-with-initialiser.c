// RUN: %clang_cc1 -emit-llvm %s -o %t

const int globalInt = 1;
int globalIntWithFloat = 1.5f;
int globalIntArray[5] = { 1, 2 };
int globalIntFromSizeOf = sizeof(globalIntArray);
char globalChar = 'a';
char globalCharArray[5] = { 'a', 'b' };
float globalFloat = 1.0f;
float globalFloatWithInt = 1;
float globalFloatArray[5] = { 1.0f, 2.0f };
double globalDouble = 1.0;
double globalDoubleArray[5] = { 1.0, 2.0 };
char *globalString = "abc";
char *globalStringArray[5] = { "123", "abc" };
long double globalLongDouble = 1;
long double globalLongDoubleArray[5] = { 1.0, 2.0 };

struct Struct {
  int member1;
  float member2;
  char *member3; 
};

struct Struct globalStruct = { 1, 2.0f, "foobar"};
