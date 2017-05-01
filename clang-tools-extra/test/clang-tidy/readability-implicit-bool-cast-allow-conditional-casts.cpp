// RUN: %check_clang_tidy %s readability-implicit-bool-cast %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: readability-implicit-bool-cast.AllowConditionalIntegerCasts, value: 1}, \
// RUN:   {key: readability-implicit-bool-cast.AllowConditionalPointerCasts, value: 1}]}' \
// RUN: -- -std=c++11

template<typename T>
void functionTaking(T);

int functionReturningInt();
int* functionReturningPointer();

struct Struct {
  int member;
};


void regularImplicitCastIntegerToBoolIsNotIgnored() {
  int integer = 0;
  functionTaking<bool>(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool [readability-implicit-bool-cast]
  // CHECK-FIXES: functionTaking<bool>(integer != 0);
}

void implicitCastIntegerToBoolInConditionalsIsAllowed() {
  if (functionReturningInt()) {}
  if (!functionReturningInt()) {}
  if (functionReturningInt() && functionReturningPointer()) {}
  if (!functionReturningInt() && !functionReturningPointer()) {}
  int value1 = functionReturningInt() ? 1 : 2;
  int value2 = !functionReturningInt() ? 1 : 2;
  int value3 = (functionReturningInt() && functionReturningPointer() || !functionReturningInt()) ? 1 : 2;
}

void regularImplicitCastPointerToBoolIsNotIgnored() {
  int* pointer = nullptr;
  functionTaking<bool>(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int *' -> bool
  // CHECK-FIXES: functionTaking<bool>(pointer != nullptr);

  int Struct::* memberPointer = &Struct::member;
  functionTaking<bool>(memberPointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int struct Struct::*' -> bool
  // CHECK-FIXES: functionTaking<bool>(memberPointer != nullptr);
}

void implicitCastPointerToBoolInConditionalsIsAllowed() {
  if (functionReturningPointer()) {}
  if (not functionReturningPointer()) {}
  int value1 = functionReturningPointer() ? 1 : 2;
  int value2 = (not functionReturningPointer()) ? 1 : 2;

  int Struct::* memberPointer = &Struct::member;
  if (memberPointer) {}
  if (memberPointer) {}
  int value3 = memberPointer ? 1 : 2;
  int value4 = (not memberPointer) ? 1 : 2;
}
