// RUN: %check_clang_tidy %s readability-use-std-min-max %t

constexpr int myConstexprMin(int a, int b) {
  return a < b ? a : b;
}

constexpr int myConstexprMax(int a, int b) {
  return a > b ? a : b;
}

int bar(int x, int y) {
  return x < y ? x : y;
}

void foo() {
  int value1,value2,value3;
  short value4;

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  if (value1 < value2)
    value1 = value2; // CHECK-FIXES: value1 = std::max(value1, value2);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  if (value1 < value2)
    value2 = value1; // CHECK-FIXES: value2 = std::min(value1, value2);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::min` instead of `>` [readability-use-std-min-max]
  if (value2 > value1)
    value2 = value1; // CHECK-FIXES: value2 = std::min(value2, value1);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::max` instead of `>` [readability-use-std-min-max
  if (value2 > value1)
    value1 = value2; // CHECK-FIXES: value1 = std::max(value2, value1);

  // No suggestion needed here
  if (value1 == value2)
    value1 = value2;
  
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  if(value1<value4)
    value1=value4; // CHECK-FIXES: value1 = std::max(value1, value4);
  
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  if(value1+value2<value3)
    value3 = value1+value2; // CHECK-FIXES: value3 = std::min(value1+value2, value3);
  
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  if (value1 < myConstexprMin(value2, value3))
    value1 = myConstexprMin(value2, value3); // CHECK-FIXES: value1 = std::max(value1, myConstexprMin(value2, value3));
  
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::min` instead of `>` [readability-use-std-min-max]
  if (value1 > myConstexprMax(value2, value3))
    value1 = myConstexprMax(value2, value3); // CHECK-FIXES: value1 = std::min(value1, myConstexprMax(value2, value3));
  
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  if (value1 < bar(value2, value3))
    value1 = bar(value2, value3); // CHECK-FIXES: value1 = std::max(value1, bar(value2, value3));
}