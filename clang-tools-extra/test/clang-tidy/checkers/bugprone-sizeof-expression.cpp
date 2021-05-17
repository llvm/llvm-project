// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- -config="{CheckOptions: [{key: bugprone-sizeof-expression.WarnOnSizeOfIntegerExpression, value: true}]}" --

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

int ReturnOverload(int) { return {}; }
S ReturnOverload(S) { return {}; }

template <class T>
T ReturnTemplate(T) { return {}; }

template <class T>
bool TestTrait1() {
  return sizeof(ReturnOverload(T{})) == sizeof(A);
}

template <class T>
bool TestTrait2() {
  return sizeof(ReturnTemplate(T{})) == sizeof(A);
}

template <class T>
bool TestTrait3() {
  return sizeof(ReturnOverload(0)) == sizeof(T{});
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
}

template <class T>
bool TestTrait4() {
  return sizeof(ReturnTemplate(0)) == sizeof(T{});
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in an integer
}

bool TestTemplates() {
  bool b = true;
  b &= TestTrait1<int>();
  b &= TestTrait1<S>();
  b &= TestTrait2<int>();
  b &= TestTrait2<S>();
  b &= TestTrait3<int>();
  b &= TestTrait3<S>();
  b &= TestTrait4<int>();
  b &= TestTrait4<S>();
  return b;
}

int Test1(const char* ptr) {
  int sum = 0;
  sum += sizeof(LEN);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(K)'
  sum += sizeof(LEN + 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(K)'
  sum += sizeof(sum, LEN);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(..., ...)'
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
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(A) / sizeof(S);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(char) / sizeof(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(char) / sizeof(A);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(B[0]) / sizeof(A);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  sum += sizeof(ptr) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'
  sum += sizeof(ptr) / sizeof(ptr[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'
  sum += sizeof(ptr) / sizeof(char*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(P*)/sizeof(Q*)'
  sum += sizeof(ptr) / sizeof(void*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(P*)/sizeof(Q*)'
  sum += sizeof(ptr) / sizeof(const void volatile*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(P*)/sizeof(Q*)'
  sum += sizeof(ptr) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'
  sum += sizeof(ptr) / sizeof(ptr[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T*)/sizeof(T)'
  sum += sizeof(int) * sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += sizeof(ptr) * sizeof(ptr[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += sizeof(int) * (2 * sizeof(char));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious 'sizeof' by 'sizeof' multiplication
  sum += (2 * sizeof(char)) * sizeof(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious 'sizeof' by 'sizeof' multiplication
  if (sizeof(A) < 0x100000) sum += 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: suspicious comparison of 'sizeof(expr)' to a constant
  if (sizeof(A) <= 0xFFFFFFFEU) sum += 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: suspicious comparison of 'sizeof(expr)' to a constant
  return sum;
}

typedef char MyChar;
typedef const MyChar MyConstChar;

int CE0 = sizeof sizeof(char);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(sizeof(...))'
int CE1 = sizeof +sizeof(char);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(sizeof(...))'
int CE2 = sizeof sizeof(const char*);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(sizeof(...))'
int CE3 = sizeof sizeof(const volatile char* const*);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(sizeof(...))'
int CE4 = sizeof sizeof(MyConstChar);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(sizeof(...))'

int Test2(MyConstChar* A) {
  int sum = 0;
  sum += sizeof(MyConstChar) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(MyConstChar) / sizeof(MyChar);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(A[0]) / sizeof(char);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  return sum;
}

template <int T>
int Foo() { int A[T]; return sizeof(T); }
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: suspicious usage of 'sizeof(K)'
template <typename T>
int Bar() { T A[5]; return sizeof(A[0]) / sizeof(T); }
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
int Test3() { return Foo<42>() + Bar<char>(); }

static const char* kABC = "abc";
static const wchar_t* kDEF = L"def";
int Test4(const char A[10]) {
  int sum = 0;
  sum += sizeof(kABC);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(char*)'
  sum += sizeof(kDEF);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(char*)'
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

  int sum = 0;
  sum += sizeof(&S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(&kGlocalMyStruct.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(&kGlocalMyStructPtr->arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(S.arr + 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(+ S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof((int*)S.arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate

  sum += sizeof(S.ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(kGlocalMyStruct.ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(kGlocalMyStructPtr->ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate

  sum += sizeof(&kGlocalMyStruct);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(&S);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(MyStruct*);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(PMyStruct);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(PS);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(PS2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(&A10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(PtrArray) / sizeof(PtrArray[1]);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(A10) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(PC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  // CHECK-MESSAGES: :[[@LINE-3]]:23: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate
  sum += sizeof(ArrayC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: suspicious usage of 'sizeof(A*)'; pointer to aggregate

  return sum;
}

int Test6() {
  int sum = 0;

  struct S A = AsStruct(), B = AsStruct();
  struct S *P = &A, *Q = &B;
  sum += sizeof(struct S) == P - Q;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += 5 * sizeof(S) != P - Q;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += sizeof(S) < P - Q;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += 5 * sizeof(S) <= P - Q;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += 5 * sizeof(*P) >= P - Q;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += Q - P > 3 * sizeof(*P);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += sizeof(S) + (P - Q);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += 5 * sizeof(S) - (P - Q);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += (P - Q) / sizeof(S);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic
  sum += (P - Q) / sizeof(*Q);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic

  return sum;
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
  sum += sizeof(&A[sizeof(A) / sizeof(int)]);
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
