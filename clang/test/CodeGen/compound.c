// RUN: %clang_cc1 < %s -emit-llvm
int A;
long long B;
int C;
int *P;
void test1(void) {
  C = (A /= B);

  P -= 4;

  C = P - (P+10);
}

short x; 
void test2(char c) { x += c; }

void foo(char *strbuf) {
  int stufflen = 4;
  strbuf += stufflen;
}


// Aggregate cast to void
union uu { int a;}; void f(union uu p) { (void) p;}

