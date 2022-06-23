// RUN: %check_clang_tidy -std=c++98 %s readability-implicit-bool-conversion %t

// We need NULL macro, but some buildbots don't like including <cstddef> header
// This is a portable way of getting it to work
#undef NULL
#define NULL 0L

template<typename T>
void functionTaking(T);

struct Struct {
  int member;
};

void useOldNullMacroInReplacements() {
  int* pointer = NULL;
  functionTaking<bool>(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int *' -> bool [readability-implicit-bool-conversion]
  // CHECK-FIXES: functionTaking<bool>(pointer != 0);

  int Struct::* memberPointer = NULL;
  functionTaking<bool>(!memberPointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit conversion 'int Struct::*' -> bool
  // CHECK-FIXES: functionTaking<bool>(memberPointer == 0);
}

void fixFalseLiteralConvertingToNullPointer() {
  functionTaking<int*>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion bool -> 'int *'
  // CHECK-FIXES: functionTaking<int*>(0);

  int* pointer = NULL;
  if (pointer == false) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: implicit conversion bool -> 'int *'
  // CHECK-FIXES: if (pointer == 0) {}

  functionTaking<int Struct::*>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion bool -> 'int Struct::*'
  // CHECK-FIXES: functionTaking<int Struct::*>(0);

  int Struct::* memberPointer = NULL;
  if (memberPointer != false) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion bool -> 'int Struct::*'
  // CHECK-FIXES: if (memberPointer != 0) {}
}
