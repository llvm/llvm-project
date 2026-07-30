// RUN: %clang_cc1 -triple=aarch64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefixes CHECK-IT,CHECK-IT-ARM
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefixes CHECK-IT,CHECK-IT-OTHER
// RUN: %clang_cc1 -triple=%ms_abi_triple -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-MS

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

int main(void) {
  int i;
// CHECK: [[I:%[a-zA-Z0-9_.]+]] = alloca i32
  // load
  i=S;
// CHECK: load i32, ptr @S
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vS;
// CHECK: load volatile i32, ptr @vS
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=*pS;
// CHECK: [[PS_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @pS
// CHECK: load i32, ptr [[PS_VAL]]
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=*pvS;
// CHECK: [[PVS_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @pvS
// CHECK: load volatile i32, ptr [[PVS_VAL]]
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=A[2];
// CHECK: load i32, ptr getelementptr {{.*}} @A
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vA[2];
// CHECK: load volatile i32, ptr getelementptr {{.*}} @vA
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=F.x;
// CHECK: load i32, ptr @F
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vF.x;
// CHECK: load volatile i32, ptr @vF
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=F2.x;
// CHECK: load i32, ptr @F2
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vF2.x;
// CHECK: load volatile i32, ptr @vF2
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vpF2->x;
// CHECK: [[VPF2_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @vpF2
// CHECK: [[ELT:%[a-zA-Z0-9_.]+]] = getelementptr {{.*}} [[VPF2_VAL]]
// CHECK: load volatile i32, ptr [[ELT]]
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=F3.x.y;
// CHECK: load i32, ptr @F3
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vF3.x.y;
// CHECK: load volatile i32, ptr @vF3
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=BF.x;
// CHECK-IT: load i8, ptr @BF
// CHECK-MS: load i32, ptr @BF
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vBF.x;
// CHECK-IT-OTHER: load volatile i8, ptr @vBF
// CHECK-IT-ARM: load volatile i32, ptr @vBF
// CHECK-MS: load volatile i32, ptr @vBF
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=V[3];
// CHECK: load <4 x i32>, ptr @V
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vV[3];
// CHECK: load volatile <4 x i32>, ptr @vV
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=VE.yx[1];
// CHECK: load <4 x i32>, ptr @VE
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vVE.zy[1];
// CHECK: load volatile <4 x i32>, ptr @vVE
// CHECK: store i32 {{.*}}, ptr [[I]]
  i = aggFct().x; // Note: not volatile
  // N.b. Aggregate return is extremely target specific, all we can
  // really say here is that there probably shouldn't be a volatile
  // load.
// CHECK-NOT: load volatile
// CHECK: store i32 {{.*}}, ptr [[I]]
  i=vtS;
// CHECK: load volatile i32, ptr @vtS
// CHECK: store i32 {{.*}}, ptr [[I]]


  // store
  S=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr @S
  vS=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr @vS
  *pS=i;
// CHECK: load i32, ptr [[I]]
// CHECK: [[PS_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @pS
// CHECK: store i32 {{.*}}, ptr [[PS_VAL]]
  *pvS=i;
// CHECK: load i32, ptr [[I]]
// CHECK: [[PVS_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @pvS
// CHECK: store volatile i32 {{.*}}, ptr [[PVS_VAL]]
  A[2]=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr getelementptr {{.*}} @A
  vA[2]=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr getelementptr {{.*}} @vA
  F.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr @F
  vF.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr @vF
  F2.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr @F2
  vF2.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr @vF2
  vpF2->x=i;
// CHECK: load i32, ptr [[I]]
// CHECK: [[VPF2_VAL:%[a-zA-Z0-9_.]+]] = load ptr, ptr @vpF2
// CHECK: [[ELT:%[a-zA-Z0-9_.]+]] = getelementptr {{.*}} [[VPF2_VAL]]
// CHECK: store volatile i32 {{.*}}, ptr [[ELT]]
  vF3.x.y=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr @vF3
  BF.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK-IT: load i8, ptr @BF
// CHECK-MS: load i32, ptr @BF
// CHECK-IT: store i8 {{.*}}, ptr @BF
// CHECK-MS: store i32 {{.*}}, ptr @BF
  vBF.x=i;
// CHECK: load i32, ptr [[I]]
// CHECK-IT-OTHER: load volatile i8, ptr @vBF
// CHECK-IT-ARM: load volatile i32, ptr @vBF
// CHECK-MS: load volatile i32, ptr @vBF
// CHECK-IT-OTHER: store volatile i8 {{.*}}, ptr @vBF
// CHECK-IT-ARM: store volatile i32 {{.*}}, ptr @vBF
// CHECK-MS: store volatile i32 {{.*}}, ptr @vBF
  V[3]=i;
// CHECK: load i32, ptr [[I]]
// CHECK: load <4 x i32>, ptr @V
// CHECK: store <4 x i32> {{.*}}, ptr @V
  vV[3]=i;
// CHECK: load i32, ptr [[I]]
// CHECK: load volatile <4 x i32>, ptr @vV
// CHECK: store volatile <4 x i32> {{.*}}, ptr @vV
  vtS=i;
// CHECK: load i32, ptr [[I]]
// CHECK: store volatile i32 {{.*}}, ptr @vtS

  // other ops:
  ++S;
// CHECK: load i32, ptr @S
// CHECK: store i32 {{.*}}, ptr @S
  ++vS;
// CHECK: load volatile i32, ptr @vS
// CHECK: store volatile i32 {{.*}}, ptr @vS
  i+=S;
// CHECK: load i32, ptr @S
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr [[I]]
  i+=vS;
// CHECK: load volatile i32, ptr @vS
// CHECK: load i32, ptr [[I]]
// CHECK: store i32 {{.*}}, ptr [[I]]
  ++vtS;
// CHECK: load volatile i32, ptr @vtS
// CHECK: store volatile i32 {{.*}}, ptr @vtS
  (void)vF2;
  // From vF2 to a temporary
// CHECK: call void @llvm.memcpy.{{.*}}(ptr align {{[0-9]+}} %{{.*}}, ptr {{.*}} @vF2, {{.*}}, i1 true)
  vF2 = vF2;
  // vF2 to itself
// CHECK: call void @llvm.memcpy.{{.*}}(ptr {{.*@vF2.*}}, ptr {{.*@vF2.*}}, i1 true)
  vF2 = vF2 = vF2;
  // vF2 to itself twice
// CHECK: call void @llvm.memcpy.{{.*}}(ptr {{.*@vF2.*}}, ptr {{.*@vF2.*}}, i1 true)
// CHECK: call void @llvm.memcpy.{{.*}}(ptr {{.*@vF2.*}}, ptr {{.*@vF2.*}}, i1 true)
  vF2 = (vF2, vF2);
  // vF2 to a temporary, then vF2 to itself
// CHECK: call void @llvm.memcpy.{{.*}}(ptr align {{[0-9]+}} %{{.*}}, ptr {{.*@vF2.*}}, i1 true)
// CHECK: call void @llvm.memcpy.{{.*}}(ptr {{.*@vF2.*}}, ptr {{.*@vF2.*}}, i1 true)
}
