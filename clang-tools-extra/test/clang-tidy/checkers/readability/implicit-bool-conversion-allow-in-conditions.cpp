// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: readability-implicit-bool-conversion.AllowIntegerConditions, value: true}, \
// RUN:   {key: readability-implicit-bool-conversion.AllowPointerConditions, value: true}]}'

template<typename T>
void functionTaking(T);

int functionReturningInt();
int* functionReturningPointer();

struct Struct {
  int member;
  unsigned bitfield : 1;
};


void regularImplicitConversionIntegerToBoolIsNotIgnored() {
  int integer = 0;
  functionTaking<bool>(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> bool [readability-implicit-bool-conversion]
  // CHECK-FIXES: functionTaking<bool>(integer != 0);
}

void implicitConversionIntegerToBoolInConditionalsIsAllowed() {
  Struct s = {};
  if (s.member) {}
  if (!s.member) {}
  if (s.bitfield) {}
  if (!s.bitfield) {}
  if (functionReturningInt()) {}
  if (!functionReturningInt()) {}
  if (functionReturningInt() && functionReturningPointer()) {}
  if (!functionReturningInt() && !functionReturningPointer()) {}
  for (; functionReturningInt(); ) {}
  for (; functionReturningPointer(); ) {}
  for (; functionReturningInt() && !functionReturningPointer() || (!functionReturningInt() && functionReturningPointer()); ) {}
  while (functionReturningInt()) {}
  while (functionReturningPointer()) {}
  while (functionReturningInt() && !functionReturningPointer() || (!functionReturningInt() && functionReturningPointer())) {}
  int value1 = functionReturningInt() ? 1 : 2;
  int value2 = !functionReturningInt() ? 1 : 2;
  int value3 = (functionReturningInt() && functionReturningPointer() || !functionReturningInt()) ? 1 : 2;
  int value4 = functionReturningInt() ?: value3;
  int *p1 = functionReturningPointer() ?: &value3;
}

void regularImplicitConversionPointerToBoolIsNotIgnored() {
  int* pointer = nullptr;
  functionTaking<bool>(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int *' -> bool
  // CHECK-FIXES: functionTaking<bool>(pointer != nullptr);

  int Struct::* memberPointer = &Struct::member;
  functionTaking<bool>(memberPointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int Struct::*' -> bool
  // CHECK-FIXES: functionTaking<bool>(memberPointer != nullptr);
}

void implicitConversionPointerToBoolInConditionalsIsAllowed() {
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
