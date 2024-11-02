// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- -config="{CheckOptions: [{key: bugprone-sizeof-expression.WarnOnSizeOfPointerToAggregate, value: false}]}" --

class C {
  int size() { return sizeof(this); }
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: suspicious usage of 'sizeof(this)'
};

#pragma pack(1)
struct  S { char a, b, c; };

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
  // No warning.
  sum += sizeof(&kGlocalMyStruct.arr);
  // No warning.
  sum += sizeof(&kGlocalMyStructPtr->arr);
  // No warning.
  sum += sizeof(S.arr + 0);
  // No warning.
  sum += sizeof(+ S.arr);
  // No warning.
  sum += sizeof((int*)S.arr);
  // No warning.

  sum += sizeof(S.ptr);
  // No warning.
  sum += sizeof(kGlocalMyStruct.ptr);
  // No warning.
  sum += sizeof(kGlocalMyStructPtr->ptr);
  // No warning.

  sum += sizeof(&kGlocalMyStruct);
  // No warning.
  sum += sizeof(&S);
  // No warning.
  sum += sizeof(MyStruct*);
  sum += sizeof(PMyStruct);
  sum += sizeof(PS);
  // No warning.
  sum += sizeof(PS2);
  // No warning.
  sum += sizeof(&A10);
  // No warning.
  sum += sizeof(PtrArray) / sizeof(PtrArray[1]);
  // No warning.
  sum += sizeof(A10) / sizeof(PtrArray[0]);
  // No warning.
  sum += sizeof(PC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of sizeof pointer 'sizeof(T)/sizeof(T)'
  sum += sizeof(ArrayC) / sizeof(PtrArray[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(...)/sizeof(...)'; numerator is not a multiple of denominator

  return sum;
}
