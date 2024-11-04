// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- -config="{CheckOptions: {bugprone-sizeof-expression.WarnOnSizeOfIntegerExpression: true, bugprone-sizeof-expression.WarnOnSizeOfPointer: true}}" --

class C {
  int size() { return sizeof(this); }
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of 'sizeof(this)'
};

#define LEN 8

int X;
extern int A[10];
extern short B[10];

#pragma pack(1)
struct  S { char a, b, c; };

enum E { E_VALUE = 0 };
enum class EC { VALUE = 0 };

bool AsBool() { return false; }
int AsInt() { return 0; }
E AsEnum() { return E_VALUE; }
EC AsEnumClass() { return EC::VALUE; }
S AsStruct() { return {}; }

struct M {
  int AsInt() { return 0; }
  E AsEnum() { return E_VALUE; }
  S AsStruct() { return {}; }
};

int Test1(const char* ptr) {
  int sum = 0;
  sum += sizeof(LEN);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(K)'
  sum += sizeof(LEN + 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(K)'
  sum += sizeof(sum, LEN);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: suspicious usage of 'sizeof(..., ...)'
  sum += sizeof(AsBool());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(AsInt());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(AsEnum());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(AsEnumClass());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(M{}.AsInt());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(M{}.AsEnum());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
  sum += sizeof(sizeof(X));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(LEN + sizeof(X));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(LEN + LEN + sizeof(X));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(LEN + (LEN + sizeof(X)));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(LEN + -sizeof(X));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(LEN + - + -sizeof(X));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))'
  sum += sizeof(char) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(A) / sizeof(S);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(char) / sizeof(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(char) / sizeof(A);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(B[0]) / sizeof(A);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(ptr) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(ptr) / sizeof(ptr[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(ptr) / sizeof(char*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(ptr) / sizeof(void*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(ptr) / sizeof(const void volatile*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(ptr) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(int) * sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += sizeof(ptr) * sizeof(ptr[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += sizeof(int) * (2 * sizeof(char));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += (2 * sizeof(char)) * sizeof(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: suspicious 'sizeof' by 'sizeof' multiplication
  if (sizeof(A) < 0x100000) sum += 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: suspicious comparison of 'sizeof(expr)' to a constant
  if (sizeof(A) <= 0xFFFFFFFEU) sum += 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: suspicious comparison of 'sizeof(expr)' to a constant
  return sum;
}

int Test5() {
  typedef int Array10[10];
  typedef C ArrayC[10];

  struct MyStruct {
    Array10 arr;
    Array10* ptr;
  };
  typedef const MyStruct TMyStruct;
  typedef const MyStruct *PMyStruct;
  typedef TMyStruct *PMyStruct2;

  static TMyStruct kGlocalMyStruct = {};
  static TMyStruct volatile * kGlocalMyStructPtr = &kGlocalMyStruct;

  MyStruct S;
  PMyStruct PS;
  PMyStruct2 PS2;
  Array10 A10;
  C *PtrArray[10];
  C *PC;

  char *PChar;
  int *PInt, **PPInt;
  MyStruct **PPMyStruct;

  int sum = 0;
  sum += sizeof(&S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(&kGlocalMyStruct.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(&kGlocalMyStructPtr->arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(S.arr + 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(+ S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof((int*)S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer

  sum += sizeof(S.ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(kGlocalMyStruct.ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(kGlocalMyStructPtr->ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer

  sum += sizeof(&kGlocalMyStruct);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(&S);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(MyStruct*);
  sum += sizeof(PMyStruct);
  sum += sizeof(PS);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PS2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(&A10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PtrArray) / sizeof(PtrArray[1]);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(A10) / sizeof(PtrArray[0]);
  sum += sizeof(PC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(ArrayC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator

  sum += sizeof(PChar);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PInt);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PPInt);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PPMyStruct);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer

  return sum;
}

void some_generic_function(const void *arg, int argsize);
int *IntP, **IntPP;
C *ClassP, **ClassPP;

void GenericFunctionTest() {
  // The `sizeof(pointer)` checks ignore situations where the pointer is
  // produced by dereferencing a pointer-to-pointer, because this is unlikely
  // to be an accident and can appear in legitimate code that tries to call
  // a generic function which emulates dynamic typing within C.
  some_generic_function(IntPP, sizeof(*IntPP));
  some_generic_function(ClassPP, sizeof(*ClassPP));
  // Using `...[0]` instead of the dereference operator is another common
  // variant, which is also widespread in the idiomatic array-size calculation:
  // `sizeof(array) / sizeof(array[0])`.
  some_generic_function(IntPP, sizeof(IntPP[0]));
  some_generic_function(ClassPP, sizeof(ClassPP[0]));
  // FIXME: There is a third common pattern where the generic function is
  // called with `&Variable` and `sizeof(Variable)`. Right now these are
  // reported by the `sizeof(pointer)` checks, but this causes some false
  // positives, so it would be good to create an exception for them.
  some_generic_function(&IntPP, sizeof(IntP));
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  some_generic_function(&ClassPP, sizeof(ClassP));
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
}

int ValidExpressions() {
  int A[] = {1, 2, 3, 4};
  static const char str[] = "hello";
  static const char* ptr[] { "aaa", "bbb", "ccc" };
  typedef C *CA10[10];
  C *PtrArray[10];
  CA10 PtrArray1;

  int sum = 0;
  if (sizeof(A) < 10)
    sum += sizeof(A);
  sum += sizeof(int);
  sum += sizeof(AsStruct());
  sum += sizeof(M{}.AsStruct());
  sum += sizeof(A[sizeof(A) / sizeof(int)]);
  // Here the outer sizeof is reported, but the inner ones are accepted:
  sum += sizeof(&A[sizeof(A) / sizeof(int)]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(sizeof(0));  // Special case: sizeof size_t.
  sum += sizeof(void*);
  sum += sizeof(void const *);
  sum += sizeof(void const *) / 4;
  sum += sizeof(str);
  sum += sizeof(str) / sizeof(char);
  sum += sizeof(str) / sizeof(str[0]);
  sum += sizeof(ptr) / sizeof(ptr[0]);
  sum += sizeof(ptr) / sizeof(*(ptr));
  sum += sizeof(PtrArray) / sizeof(PtrArray[0]);
  // Canonical type of PtrArray1 is same as PtrArray.
  sum = sizeof(PtrArray) / sizeof(PtrArray1[0]);
  // There is no warning for 'sizeof(T*)/sizeof(Q)' case.
  sum += sizeof(PtrArray) / sizeof(A[0]);
  return sum;
}
