// RUN: %check_clang_tidy -std=c++11-or-later %s readability-use-std-min-max %t -- -- -fno-delayed-template-parsing
#define MY_MACRO_MIN(a, b) ((a) < (b) ? (a) : (b))

constexpr int myConstexprMin(int a, int b) {
  return a < b ? a : b;
}

constexpr int myConstexprMax(int a, int b) {
  return a > b ? a : b;
}

#define MY_IF_MACRO(condition, statement) \
  if (condition) {                        \
    statement                             \
  }                                       

class MyClass {
public:
  int member1;
  int member2;
};

template<typename T>

void foo(T value7) {
  int value1,value2,value3;

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value1, value2);
  if (value1 < value2)
    value1 = value2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value2 = std::min(value1, value2);
  if (value1 < value2)
    value2 = value1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: value2 = std::min(value2, value1);
  if (value2 > value1)
    value2 = value1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value2, value1);
  if (value2 > value1)
    value1 = value2; 

  // No suggestion needed here
  if (value1 == value2)
    value1 = value2;

  // CHECK-MESSAGES: :[[@LINE+3]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max<int>(value1, value4);
  short value4;
  if(value1<value4)
    value1=value4; 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value3 = std::min(value1+value2, value3);
  if(value1+value2<value3)
    value3 = value1+value2; 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value1, myConstexprMin(value2, value3));
  if (value1 < myConstexprMin(value2, value3))
    value1 = myConstexprMin(value2, value3); 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::min(value1, myConstexprMax(value2, value3));
  if (value1 > myConstexprMax(value2, value3))
    value1 = myConstexprMax(value2, value3); 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<=` [readability-use-std-min-max]
  // CHECK-FIXES: value2 = std::min(value1, value2);
  if (value1 <= value2)
    value2 = value1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<=` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value1, value2);
  if (value1 <= value2)
    value1 = value2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `>=` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value2, value1);
  if (value2 >= value1)
    value1 = value2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `>=` [readability-use-std-min-max]
  // CHECK-FIXES: value2 = std::min(value2, value1);
  if (value2 >= value1)
    value2 = value1; 
  
  // CHECK-MESSAGES: :[[@LINE+3]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member1 = std::max(obj.member1, obj.member2);
  MyClass obj;
  if (obj.member1 < obj.member2)
    obj.member1 = obj.member2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member2 = std::min(obj.member1, obj.member2);
  if (obj.member1 < obj.member2)
    obj.member2 = obj.member1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member2 = std::min(obj.member2, obj.member1);
  if (obj.member2 > obj.member1)
    obj.member2 = obj.member1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member1 = std::max(obj.member2, obj.member1);
  if (obj.member2 > obj.member1)
    obj.member1 = obj.member2; 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member1 = std::max<int>(obj.member1, value4);
  if (obj.member1 < value4)
    obj.member1 = value4; 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value3 = std::min(obj.member1 + value2, value3);
  if (obj.member1 + value2 < value3)
    value3 = obj.member1 + value2; 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<=` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member2 = std::min(value1, obj.member2);
  if (value1 <= obj.member2)
    obj.member2 = value1; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<=` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value1, obj.member2);
  if (value1 <= obj.member2)
    value1 = obj.member2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `>=` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(obj.member2, value1);
  if (obj.member2 >= value1)
    value1 = obj.member2; 

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `>=` [readability-use-std-min-max]
  // CHECK-FIXES: obj.member2 = std::min(obj.member2, value1);
  if (obj.member2 >= value1)
    obj.member2 = value1; 
  
  // No suggestion needed here
  if (MY_MACRO_MIN(value1, value2) < value3)
    value3 = MY_MACRO_MIN(value1, value2); 
  
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value4 = std::max<int>(value4, value2);
  if (value4 < value2){
    value4 = value2; 
  }

  // No suggestion needed here
  if(value1 < value2)
    value2 = value1;
  else
    value2 = value3;
  
  // No suggestion needed here
  if(value1<value2){
    value2 = value1; 
  }
  else{
    value2 = value3;  
  }

  // No suggestion needed here
  if(value1<value2){
    value2 = value1; 
    int res = value1 + value2;
  }

  // No suggestion needed here
  MY_IF_MACRO(value1 < value2, value1 = value2;)

  // No suggestion needed here
  if(value1<value2){
    value1 = value2;
  }
  else if(value1>value2){
    value2 = value1;
  }

  // CHECK-MESSAGES: :[[@LINE+3]]:5: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max(value1, value3);
  if(value1 == value2){
    if(value1<value3)
      value1 = value3;
  }

  // CHECK-MESSAGES: :[[@LINE+5]]:7: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value1 = std::max<int>(value1, value4);
  if(value1 == value2){
    if(value2 == value3){
      value3+=1;
      if(value1<value4){
        value1 = value4;
      }
    }
    else if(value3>value2){
      value2 = value3;
    }
  }
  
  // CHECK-MESSAGES: :[[@LINE+4]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value6 = std::min<unsigned int>(value5, value6);
  unsigned int value5;
  unsigned char value6;
  if(value5<value6){
    value6 = value5;
  }

  //No suggestion needed here
  if(value7<value6){
    value6 = value7;
  }

  //CHECK-MESSAGES: :[[@LINE+3]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  //CHECK-FIXES: value1 = std::min(value8, value1);
  const int value8 = 5;
  if(value8<value1)
    value1 = value8;
  
  //CHECK-MESSAGES: :[[@LINE+3]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  //CHECK-FIXES: value1 = std::min(value9, value1);
  volatile int value9 = 6;
  if(value9<value1)
    value1 = value9;
}

using my_size = unsigned long long;

template<typename T>
struct MyVector
{
    using size_type = my_size;
    size_type size() const;
};

void testVectorSizeType() {
  MyVector<int> v;
  unsigned int value;

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `>` [readability-use-std-min-max]
  // CHECK-FIXES: value = std::max<my_size>(v.size(), value);
  if (v.size() > value)
    value = v.size();

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: value = std::max<my_size>(value, v.size());
  if (value < v.size())
    value = v.size();
}

namespace gh121676 {

void useLeft() {
  using U16 = unsigned short;
  U16 I = 0;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: I = std::max<U16>(I, 16U);
  if (I < 16U)
    I = 16U;
}
void useRight() {
  using U16 = unsigned short;
  U16 I = 0;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use `std::min` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: I = std::min<U16>(16U, I);
  if (16U < I)
    I = 16U;
}

} // namespace gh121676
