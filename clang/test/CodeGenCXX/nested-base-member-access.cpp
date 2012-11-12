// RUN: %clang_cc1 %s -emit-llvm -o %t

extern "C" int printf(...);

struct M {
  M(int i){ iM = i; }
  int iM;
  void MPR() { printf("iM = %d\n", iM); }

};

struct Q {
  Q(int i){ iQ = i; }
  int iQ;
  void QPR() { printf("iQ = %d\n", iQ); }
};

struct IQ {
  IQ(int i) { iIQ = i; }
  void IQPR() { printf("iIQ = %d\n", iIQ); }
  int iIQ;
};

struct L : IQ {
  L(int i) : IQ(i+100) { iL = i; }
  int iL;
};

struct P : Q, L  {
  P(int i) : Q(i+100), L(i+200) { iP = i; }
  int iP;
  void PPR() { printf("iP = %d\n", iP); }
};


struct N : M,P {
  N() : M(100), P(200) {}
  void PR() {
    this->MPR(); this->PPR(); this->QPR(); 
    IQPR();
    printf("iM = %d\n", iM); 
    printf("iP = %d\n", iP);
    printf("iQ = %d\n", iQ);
    printf("iL = %d\n", iL);
    printf("iIQ = %d\n", iIQ);
  }
};

int main() {
  N n1;
  n1.PR();
}
