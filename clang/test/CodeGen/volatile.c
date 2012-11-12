// RUN: %clang_cc1 -emit-llvm < %s -o %t
// RUN: grep volatile %t | count 28
// RUN: grep memcpy %t | count 7

// The number 28 comes from the current codegen for volatile loads;
// if this number changes, it's not necessarily something wrong, but
// something has changed to affect volatile load/store codegen

int S;
volatile int vS;

int* pS;
volatile int* pvS;

int A[10];
volatile int vA[10];

struct { int x; } F;
struct { volatile int x; } vF;

struct { int x; } F2;
volatile struct { int x; } vF2;
volatile struct { int x; } *vpF2;

struct { struct { int y; } x; } F3;
volatile struct { struct { int y; } x; } vF3;

struct { int x:3; } BF;
struct { volatile int x:3; } vBF;

typedef int v4si __attribute__ ((vector_size (16)));
v4si V;
volatile v4si vV;

typedef __attribute__(( ext_vector_type(4) )) int extv4;
extv4 VE;
volatile extv4 vVE;

volatile struct {int x;} aggFct(void);

typedef volatile int volatile_int;
volatile_int vtS;

int main() {
  int i;

  // load
  i=S;
  i=vS;
  i=*pS;
  i=*pvS;
  i=A[2];
  i=vA[2];
  i=F.x;
  i=vF.x;
  i=F2.x;
  i=vF2.x;
  i=vpF2->x;
  i=F3.x.y;
  i=vF3.x.y;
  i=BF.x;
  i=vBF.x;
  i=V[3];
  i=vV[3];
  i=VE.yx[1];
  i=vVE.zy[1];
  i = aggFct().x; // Note: not volatile
  i=vtS;


  // store
  S=i;
  vS=i;
  *pS=i;
  *pvS=i;
  A[2]=i;
  vA[2]=i;
  F.x=i;
  vF.x=i;
  F2.x=i;
  vF2.x=i;
  vpF2->x=i;
  vF3.x.y=i;
  BF.x=i;
  vBF.x=i;
  V[3]=i;
  vV[3]=i;
  vtS=i;

  // other ops:
  ++S;
  ++vS;
  i+=S;
  i+=vS;
  ++vtS;
  (void)vF2;
  vF2 = vF2;
  vF2 = vF2 = vF2;
  vF2 = (vF2, vF2);
}
