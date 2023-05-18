// RUN: %check_clang_tidy %s bugprone-multi-level-implicit-pointer-conversion %t

using OneStar = void*;
using OneStarFancy = OneStar;

void takeFirstLevelVoidPtr(OneStar message);
void takeFirstLevelConstVoidPtr(const OneStarFancy message);
void takeFirstLevelConstVoidPtrConst(const void* const message);
void takeSecondLevelVoidPtr(void** message);

void** getSecondLevelVoidPtr();
void* getFirstLevelVoidPtr();
int** getSecondLevelIntPtr();
int* getFirstLevelIntPtr();

int table[5];

void test()
{
  void** secondLevelVoidPtr;
  int* firstLevelIntPtr;

  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: multilevel pointer conversion from 'void **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  void* a = getSecondLevelVoidPtr();

  void** b = getSecondLevelVoidPtr();
  void* c = getFirstLevelVoidPtr();

  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: multilevel pointer conversion from 'int **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  void* d = getSecondLevelIntPtr();

  takeFirstLevelVoidPtr(&table);

  takeFirstLevelVoidPtr(firstLevelIntPtr);

  takeFirstLevelVoidPtr(getFirstLevelIntPtr());

  // CHECK-MESSAGES: :[[@LINE+1]]:25: warning: multilevel pointer conversion from 'void **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelVoidPtr(secondLevelVoidPtr);

  // CHECK-MESSAGES: :[[@LINE+1]]:30: warning: multilevel pointer conversion from 'void **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelConstVoidPtr(secondLevelVoidPtr);

  // CHECK-MESSAGES: :[[@LINE+1]]:35: warning: multilevel pointer conversion from 'void **' to 'const void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelConstVoidPtrConst(secondLevelVoidPtr);

  // CHECK-MESSAGES: :[[@LINE+1]]:35: warning: multilevel pointer conversion from 'void ***' to 'const void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelConstVoidPtrConst(&secondLevelVoidPtr);

  takeSecondLevelVoidPtr(secondLevelVoidPtr);

  // CHECK-MESSAGES: :[[@LINE+1]]:25: warning: multilevel pointer conversion from 'void **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelVoidPtr(getSecondLevelVoidPtr());

  // CHECK-MESSAGES: :[[@LINE+1]]:30: warning: multilevel pointer conversion from 'void **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelConstVoidPtr(getSecondLevelVoidPtr());

  // CHECK-MESSAGES: :[[@LINE+1]]:35: warning: multilevel pointer conversion from 'void **' to 'const void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelConstVoidPtrConst(getSecondLevelVoidPtr());

  // CHECK-MESSAGES: :[[@LINE+1]]:25: warning: multilevel pointer conversion from 'int **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  takeFirstLevelVoidPtr(getSecondLevelIntPtr());

  takeSecondLevelVoidPtr(getSecondLevelVoidPtr());
}
