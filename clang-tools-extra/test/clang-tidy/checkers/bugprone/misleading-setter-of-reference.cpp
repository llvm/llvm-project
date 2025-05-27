// RUN: %check_clang_tidy %s bugprone-misleading-setter-of-reference %t

struct X {
  X &operator=(const X &) { return *this; }
private:
  int &Mem;
  friend class Test1;
};

class Test1 {
  X &MemX;
  int &MemI;
protected:
  long &MemL;
public:
  long &MemLPub;

  Test1(X &MemX, int &MemI, long &MemL) : MemX(MemX), MemI(MemI), MemL(MemL), MemLPub(MemL) {}
  void setI(int *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setI' can be mistakenly used in order to change the reference 'MemI' instead of the value of it
    MemI = *NewValue;
  }
  void setL(long *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setL' can be mistakenly used in order to change the reference 'MemL' instead of the value of it
    MemL = *NewValue;
  }
  void setX(X *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setX' can be mistakenly used in order to change the reference 'MemX' instead of the value of it
    MemX = *NewValue;
  }
  void set1(int *NewValue) {
    MemX.Mem = *NewValue;
  }
  void set2(int *NewValue) {
    MemL = static_cast<long>(*NewValue);
  }
  void set3(int *NewValue) {
    MemI = *NewValue;
    MemL = static_cast<long>(*NewValue);
  }
  void set4(long *NewValue, int) {
    MemL = *NewValue;
  }
  void setLPub(long *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setLPub' can be mistakenly used in order to change the reference 'MemLPub' instead of the value of it
    MemLPub = *NewValue;
  }

private:
  void set5(long *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'set5' can be mistakenly used in order to change the reference 'MemL' instead of the value of it
    MemL = *NewValue;
  }
};

class Base {
protected:
  int &MemI;
};

class Derived : public Base {
public:
  void setI(int *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setI' can be mistakenly used in order to change the reference 'MemI' instead of the value of it
    MemI = *NewValue;
  }
};

using UIntRef = unsigned int &;
using UIntPtr = unsigned int *;
using UInt = unsigned int;

class AliasTest {
  UIntRef Value1;
  UInt &Value2;
  unsigned int &Value3;
public:
  void setValue1(UIntPtr NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setValue1' can be mistakenly used in order to change the reference 'Value1' instead of the value of it
    Value1 = *NewValue;
  }
  void setValue2(unsigned int *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setValue2' can be mistakenly used in order to change the reference 'Value2' instead of the value of it
    Value2 = *NewValue;
  }
  void setValue3(UInt *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setValue3' can be mistakenly used in order to change the reference 'Value3' instead of the value of it
    Value3 = *NewValue;
  }
};

template <typename T>
class TemplateTest {
  T &Mem;
public:
  TemplateTest(T &V) : Mem{V} {}
  void setValue(T *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setValue' can be mistakenly used in order to change the reference 'Mem' instead of the value of it
    Mem = *NewValue;
  }
};

void f_TemplateTest(char *Value) {
  char CharValue;
  TemplateTest<char> TTChar{CharValue};
  TTChar.setValue(Value);
}

template <typename T>
class AddMember {
protected:
  T &Value;
};

class TemplateBaseTest : public AddMember<int> {
public:
  void setValue(int *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setValue' can be mistakenly used in order to change the reference 'Value' instead of the value of it
    Value = *NewValue;
  }
};
