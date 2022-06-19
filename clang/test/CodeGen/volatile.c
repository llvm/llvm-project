// RUN: %clang_cc1 -no-opaque-pointers -triple=aarch64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefixes CHECK-IT,CHECK-IT-ARM
// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefixes CHECK-IT,CHECK-IT-OTHER
// RUN: %clang_cc1 -no-opaque-pointers -triple=%ms_abi_triple -emit-llvm < %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-MS

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
// CHECK: load i32, i32* @S
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vS;
// CHECK: load volatile i32, i32* @vS
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=*pS;
// CHECK: [[PS_VAL:%[a-zA-Z0-9_.]+]] = load i32*, i32** @pS
// CHECK: load i32, i32* [[PS_VAL]]
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=*pvS;
// CHECK: [[PVS_VAL:%[a-zA-Z0-9_.]+]] = load i32*, i32** @pvS
// CHECK: load volatile i32, i32* [[PVS_VAL]]
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=A[2];
// CHECK: load i32, i32* getelementptr {{.*}} @A
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vA[2];
// CHECK: load volatile i32, i32* getelementptr {{.*}} @vA
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=F.x;
// CHECK: load i32, i32* getelementptr {{.*}} @F
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vF.x;
// CHECK: load volatile i32, i32* getelementptr {{.*}} @vF
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=F2.x;
// CHECK: load i32, i32* getelementptr {{.*}} @F2
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vF2.x;
// CHECK: load volatile i32, i32* getelementptr {{.*}} @vF2
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vpF2->x;
// CHECK: [[VPF2_VAL:%[a-zA-Z0-9_.]+]] = load {{%[a-zA-Z0-9_.]+}}*, {{%[a-zA-Z0-9_.]+}}** @vpF2
// CHECK: [[ELT:%[a-zA-Z0-9_.]+]] = getelementptr {{.*}} [[VPF2_VAL]]
// CHECK: load volatile i32, i32* [[ELT]]
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=F3.x.y;
// CHECK: load i32, i32* getelementptr {{.*}} @F3
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vF3.x.y;
// CHECK: load volatile i32, i32* getelementptr {{.*}} @vF3
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=BF.x;
// CHECK-IT: load i8, i8* getelementptr {{.*}} @BF
// CHECK-MS: load i32, i32* getelementptr {{.*}} @BF
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vBF.x;
// CHECK-IT-OTHER: load volatile i8, i8* getelementptr {{.*}} @vBF
// CHECK-IT-ARM: load volatile i32, i32* bitcast {{.*}} @vBF
// CHECK-MS: load volatile i32, i32* getelementptr {{.*}} @vBF
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=V[3];
// CHECK: load <4 x i32>, <4 x i32>* @V
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vV[3];
// CHECK: load volatile <4 x i32>, <4 x i32>* @vV
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=VE.yx[1];
// CHECK: load <4 x i32>, <4 x i32>* @VE
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vVE.zy[1];
// CHECK: load volatile <4 x i32>, <4 x i32>* @vVE
// CHECK: store i32 {{.*}}, i32* [[I]]
  i = aggFct().x; // Note: not volatile
  // N.b. Aggregate return is extremely target specific, all we can
  // really say here is that there probably shouldn't be a volatile
  // load.
// CHECK-NOT: load volatile
// CHECK: store i32 {{.*}}, i32* [[I]]
  i=vtS;
// CHECK: load volatile i32, i32* @vtS
// CHECK: store i32 {{.*}}, i32* [[I]]


  // store
  S=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* @S
  vS=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* @vS
  *pS=i;
// CHECK: load i32, i32* [[I]]
// CHECK: [[PS_VAL:%[a-zA-Z0-9_.]+]] = load i32*, i32** @pS
// CHECK: store i32 {{.*}}, i32* [[PS_VAL]]
  *pvS=i;
// CHECK: load i32, i32* [[I]]
// CHECK: [[PVS_VAL:%[a-zA-Z0-9_.]+]] = load i32*, i32** @pvS
// CHECK: store volatile i32 {{.*}}, i32* [[PVS_VAL]]
  A[2]=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* getelementptr {{.*}} @A
  vA[2]=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* getelementptr {{.*}} @vA
  F.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* getelementptr {{.*}} @F
  vF.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* getelementptr {{.*}} @vF
  F2.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* getelementptr {{.*}} @F2
  vF2.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* getelementptr {{.*}} @vF2
  vpF2->x=i;
// CHECK: load i32, i32* [[I]]
// CHECK: [[VPF2_VAL:%[a-zA-Z0-9_.]+]] = load {{%[a-zA-Z0-9._]+}}*, {{%[a-zA-Z0-9._]+}}** @vpF2
// CHECK: [[ELT:%[a-zA-Z0-9_.]+]] = getelementptr {{.*}} [[VPF2_VAL]]
// CHECK: store volatile i32 {{.*}}, i32* [[ELT]]
  vF3.x.y=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* getelementptr {{.*}} @vF3
  BF.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK-IT: load i8, i8* getelementptr {{.*}} @BF
// CHECK-MS: load i32, i32* getelementptr {{.*}} @BF
// CHECK-IT: store i8 {{.*}}, i8* getelementptr {{.*}} @BF
// CHECK-MS: store i32 {{.*}}, i32* getelementptr {{.*}} @BF
  vBF.x=i;
// CHECK: load i32, i32* [[I]]
// CHECK-IT-OTHER: load volatile i8, i8* getelementptr {{.*}} @vBF
// CHECK-IT-ARM: load volatile i32, i32* bitcast {{.*}} @vBF
// CHECK-MS: load volatile i32, i32* getelementptr {{.*}} @vBF
// CHECK-IT-OTHER: store volatile i8 {{.*}}, i8* getelementptr {{.*}} @vBF
// CHECK-IT-ARM: store volatile i32 {{.*}}, i32* bitcast {{.*}} @vBF
// CHECK-MS: store volatile i32 {{.*}}, i32* getelementptr {{.*}} @vBF
  V[3]=i;
// CHECK: load i32, i32* [[I]]
// CHECK: load <4 x i32>, <4 x i32>* @V
// CHECK: store <4 x i32> {{.*}}, <4 x i32>* @V
  vV[3]=i;
// CHECK: load i32, i32* [[I]]
// CHECK: load volatile <4 x i32>, <4 x i32>* @vV
// CHECK: store volatile <4 x i32> {{.*}}, <4 x i32>* @vV
  vtS=i;
// CHECK: load i32, i32* [[I]]
// CHECK: store volatile i32 {{.*}}, i32* @vtS

  // other ops:
  ++S;
// CHECK: load i32, i32* @S
// CHECK: store i32 {{.*}}, i32* @S
  ++vS;
// CHECK: load volatile i32, i32* @vS
// CHECK: store volatile i32 {{.*}}, i32* @vS
  i+=S;
// CHECK: load i32, i32* @S
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* [[I]]
  i+=vS;
// CHECK: load volatile i32, i32* @vS
// CHECK: load i32, i32* [[I]]
// CHECK: store i32 {{.*}}, i32* [[I]]
  ++vtS;
// CHECK: load volatile i32, i32* @vtS
// CHECK: store volatile i32 {{.*}}, i32* @vtS
  (void)vF2;
  // From vF2 to a temporary
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align {{[0-9]+}} %{{.*}}, i8* {{.*}} @vF2 {{.*}}, i1 true)
  vF2 = vF2;
  // vF2 to itself
// CHECK: call void @llvm.memcpy.{{.*}}(i8* {{.*@vF2.*}}, i8* {{.*@vF2.*}}, i1 true)
  vF2 = vF2 = vF2;
  // vF2 to itself twice
// CHECK: call void @llvm.memcpy.{{.*}}(i8* {{.*@vF2.*}}, i8* {{.*@vF2.*}}, i1 true)
// CHECK: call void @llvm.memcpy.{{.*}}(i8* {{.*@vF2.*}}, i8* {{.*@vF2.*}}, i1 true)
  vF2 = (vF2, vF2);
  // vF2 to a temporary, then vF2 to itself
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align {{[0-9]+}} %{{.*}}, i8* {{.*@vF2.*}}, i1 true)
// CHECK: call void @llvm.memcpy.{{.*}}(i8* {{.*@vF2.*}}, i8* {{.*@vF2.*}}, i1 true)
}
