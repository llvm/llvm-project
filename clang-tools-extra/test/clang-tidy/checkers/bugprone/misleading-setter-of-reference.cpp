// RUN: %check_clang_tidy %s bugprone-misleading-setter-of-reference %t

struct X {
  X &operator=(const X &) { return *this; }
  int &Mem;
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
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setI' can be mistakenly used in order to change the reference 'MemI' instead of the value of it [bugprone-misleading-setter-of-reference]
    MemI = *NewValue;
  }
  void setL(long *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setL' can be mistakenly used in order to change the reference 'MemL' instead of the value of it [bugprone-misleading-setter-of-reference]
    MemL = *NewValue;
  }
  void setX(X *NewValue) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'setX' can be mistakenly used in order to change the reference 'MemX' instead of the value of it [bugprone-misleading-setter-of-reference]
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
    MemLPub = *NewValue;
  }

protected:
  void set5(long *NewValue) {
    MemL = *NewValue;
  }
};
