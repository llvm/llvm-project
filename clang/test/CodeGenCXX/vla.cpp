// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck -check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -std=c++11 -triple amdgcn %s -emit-llvm -o - | FileCheck -check-prefixes=AMDGCN,CHECK %s

template<typename T>
struct S {
  static int n;
};
template<typename T> int S<T>::n = 5;

int f() {
  // Make sure that the reference here is enough to trigger the instantiation of
  // the static data member.
  // CHECK: @_ZN1SIiE1nE = linkonce_odr{{.*}} global i32 5
  int a[S<int>::n];
  return sizeof a;
}

// rdar://problem/9506377
void test0(void *array, int n) {
  // CHECK-LABEL: define void @_Z5test0Pvi(
  // X64:        [[ARRAY:%.*]] = alloca i8*, align 8
  // AMDGCN:        [[ARRAY0:%.*]] = alloca i8*, align 8, addrspace(5)
  // AMDGCN-NEXT:   [[ARRAY:%.*]] = addrspacecast i8* addrspace(5)* [[ARRAY0]] to i8**
  // X64-NEXT:   [[N:%.*]] = alloca i32, align 4
  // AMDGCN:        [[N0:%.*]] = alloca i32, align 4, addrspace(5)
  // AMDGCN-NEXT:   [[N:%.*]] = addrspacecast i32 addrspace(5)* [[N0]] to i32*
  // X64-NEXT:   [[REF:%.*]] = alloca i16*, align 8
  // AMDGCN:        [[REF0:%.*]] = alloca i16*, align 8, addrspace(5)
  // AMDGCN-NEXT:   [[REF:%.*]] = addrspacecast i16* addrspace(5)* [[REF0]] to i16**
  // X64-NEXT:   [[S:%.*]] = alloca i16, align 2
  // AMDGCN:        [[S0:%.*]] = alloca i16, align 2, addrspace(5)
  // AMDGCN-NEXT:   [[S:%.*]] = addrspacecast i16 addrspace(5)* [[S0]] to i16*
  // CHECK-NEXT: store i8* 
  // CHECK-NEXT: store i32

  // Capture the bounds.
  // CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[N]], align 4
  // CHECK-NEXT: [[DIM0:%.*]] = zext i32 [[T0]] to i64
  // CHECK-NEXT: [[T0:%.*]] = load i32, i32* [[N]], align 4
  // CHECK-NEXT: [[T1:%.*]] = add nsw i32 [[T0]], 1
  // CHECK-NEXT: [[DIM1:%.*]] = zext i32 [[T1]] to i64
  typedef short array_t[n][n+1];

  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[ARRAY]], align 8
  // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[T0]] to i16*
  // CHECK-NEXT: store i16* [[T1]], i16** [[REF]], align 8
  array_t &ref = *(array_t*) array;

  // CHECK-NEXT: [[T0:%.*]] = load i16*, i16** [[REF]]
  // CHECK-NEXT: [[T1:%.*]] = mul nsw i64 1, [[DIM1]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i16, i16* [[T0]], i64 [[T1]]
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i16, i16* [[T2]], i64 2
  // CHECK-NEXT: store i16 3, i16* [[T3]]
  ref[1][2] = 3;

  // CHECK-NEXT: [[T0:%.*]] = load i16*, i16** [[REF]]
  // CHECK-NEXT: [[T1:%.*]] = mul nsw i64 4, [[DIM1]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i16, i16* [[T0]], i64 [[T1]]
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i16, i16* [[T2]], i64 5
  // CHECK-NEXT: [[T4:%.*]] = load i16, i16* [[T3]]
  // CHECK-NEXT: store i16 [[T4]], i16* [[S]], align 2
  short s = ref[4][5];

  // CHECK-NEXT: ret void
}


void test2(int b) {
  // CHECK-LABEL: define void {{.*}}test2{{.*}}(i32 %b)
  int varr[b];
  // AMDGCN: %__end1 = alloca i32*, align 8, addrspace(5)
  // AMDGCN: [[END:%.*]] = addrspacecast i32* addrspace(5)* %__end1 to i32**
  // get the address of %b by checking the first store that stores it 
  //CHECK: store i32 %b, i32* [[PTR_B:%.*]]

  // get the size of the VLA by getting the first load of the PTR_B
  //CHECK: [[VLA_NUM_ELEMENTS_PREZEXT:%.*]] = load i32, i32* [[PTR_B]]
  //CHECK-NEXT: [[VLA_NUM_ELEMENTS_PRE:%.*]] = zext i32 [[VLA_NUM_ELEMENTS_PREZEXT]]
  
  b = 15;
  //CHECK: store i32 15, i32* [[PTR_B]]
  
  // Now get the sizeof, and then divide by the element size
  
  
  //CHECK: [[VLA_SIZEOF:%.*]] = mul nuw i64 4, [[VLA_NUM_ELEMENTS_PRE]]
  //CHECK-NEXT: [[VLA_NUM_ELEMENTS_POST:%.*]] = udiv i64 [[VLA_SIZEOF]], 4
  //CHECK-NEXT: [[VLA_END_PTR:%.*]] = getelementptr inbounds i32, i32* {{%.*}}, i64 [[VLA_NUM_ELEMENTS_POST]]
  //X64-NEXT: store i32* [[VLA_END_PTR]], i32** %__end1
  //AMDGCN-NEXT: store i32* [[VLA_END_PTR]], i32** [[END]]
  for (int d : varr) 0;
}

void test3(int b, int c) {
  // CHECK-LABEL: define void {{.*}}test3{{.*}}(i32 %b, i32 %c)
  int varr[b][c];
  // AMDGCN: %__end1 = alloca i32*, align 8, addrspace(5)
  // AMDGCN: [[END:%.*]] = addrspacecast i32* addrspace(5)* %__end1 to i32**
  // get the address of %b by checking the first store that stores it 
  //CHECK: store i32 %b, i32* [[PTR_B:%.*]]
  //CHECK-NEXT: store i32 %c, i32* [[PTR_C:%.*]]
  
  // get the size of the VLA by getting the first load of the PTR_B
  //CHECK: [[VLA_DIM1_PREZEXT:%.*]] = load i32, i32* [[PTR_B]]
  //CHECK-NEXT: [[VLA_DIM1_PRE:%.*]] = zext i32 [[VLA_DIM1_PREZEXT]]
  //CHECK: [[VLA_DIM2_PREZEXT:%.*]] = load i32, i32* [[PTR_C]]
  //CHECK-NEXT: [[VLA_DIM2_PRE:%.*]] = zext i32 [[VLA_DIM2_PREZEXT]]
  
  b = 15;
  c = 15;
  //CHECK: store i32 15, i32* [[PTR_B]]
  //CHECK: store i32 15, i32* [[PTR_C]]
  // Now get the sizeof, and then divide by the element size
  
  // multiply the two dimensions, then by the element type and then divide by the sizeof dim2
  //CHECK: [[VLA_DIM1_X_DIM2:%.*]] = mul nuw i64 [[VLA_DIM1_PRE]], [[VLA_DIM2_PRE]]
  //CHECK-NEXT: [[VLA_SIZEOF:%.*]] = mul nuw i64 4, [[VLA_DIM1_X_DIM2]]
  //CHECK-NEXT: [[VLA_SIZEOF_DIM2:%.*]] = mul nuw i64 4, [[VLA_DIM2_PRE]]
  //CHECK-NEXT: [[VLA_NUM_ELEMENTS:%.*]] = udiv i64 [[VLA_SIZEOF]], [[VLA_SIZEOF_DIM2]]
  //CHECK-NEXT: [[VLA_END_INDEX:%.*]] = mul nsw i64 [[VLA_NUM_ELEMENTS]], [[VLA_DIM2_PRE]]
  //CHECK-NEXT: [[VLA_END_PTR:%.*]] = getelementptr inbounds i32, i32* {{%.*}}, i64 [[VLA_END_INDEX]]
  //X64-NEXT: store i32* [[VLA_END_PTR]], i32** %__end
  //AMDGCN-NEXT: store i32* [[VLA_END_PTR]], i32** [[END]]
 
  for (auto &d : varr) 0;
}


