// RUN: %clang_cc1 %s -emit-llvm -o -

union u_tag {
  int a;
  float b;
} u;

void f() {
  u.b = 11;
}

float get_b(union u_tag *my_u) {
  return my_u->b;
}

int f2( float __x ) { 
  union{ 
    float __f; 
    unsigned int __u; 
  }__u;
  return (int)(__u.__u >> 31); 
}

typedef union { int i; int *j; } value;

int f3(value v) {
  return *v.j;
}

enum E9 { one, two };
union S65 { enum E9 a; } ; union S65 s65;
void fS65() { enum E9 e = s65.a; } 

typedef union{
  unsigned char x[65536];
} q;
int qfunc() {q buf; unsigned char* x = buf.x;}

union RR {_Bool a : 1;} RRU;
int RRF(void) {return RRU.a;}

// PR6164
typedef union T0 { unsigned int : 0; } T0;
T0 t0;

union { int large_bitfield: 31; char c } u2;
