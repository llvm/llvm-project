// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- --
// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t -- -- -x c++

#ifdef __cplusplus
#define STRKWD
#else
#define STRKWD struct
#endif

int Test5() {
  typedef int Array10[10];

  struct MyStruct {
    Array10 arr;
    Array10* ptr;
  };

  typedef struct TypedefStruct {
    Array10 arr;
    Array10* ptr;
  } TypedefStruct;

  typedef const STRKWD MyStruct TMyStruct;
  typedef const STRKWD MyStruct *PMyStruct;
  typedef TMyStruct *PMyStruct2;
  typedef const TypedefStruct *PTTStruct;

  STRKWD MyStruct S;
  TypedefStruct TS;
  PMyStruct PS;
  PMyStruct2 PS2;
  Array10 A10;
  PTTStruct PTTS;

  int sum = 0;
  sum += sizeof(&S);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(__typeof(&S));
  sum += sizeof(&TS);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(__typeof(&TS));
  sum += sizeof(STRKWD MyStruct*);
  sum += sizeof(__typeof(STRKWD MyStruct*));
  sum += sizeof(TypedefStruct*);
  sum += sizeof(__typeof(TypedefStruct*));
  sum += sizeof(PTTS);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PMyStruct);
  sum += sizeof(PS);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(PS2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer
  sum += sizeof(&A10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof()' on an expression that results in a pointer

#ifdef __cplusplus
  MyStruct &rS = S;
  sum += sizeof(rS); // same as sizeof(S), not a pointer.  So should not warn.
#endif

  return sum;
}
