// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O0 -o - | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -O0 -o - | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,INLINE

// Tests that user functions will always be inlined.
// This includes exported functions and mangled entry point implementation functions.
// The unmangled entry functions must not be alwaysinlined.

#define MAX 100

float nums[MAX];

// Verify that all functions have the alwaysinline attribute
// NOINLINE: Function Attrs: alwaysinline
// NOINLINE: define void @_Z4swapA100_jjj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %ix1, i32 noundef %ix2) [[IntAttr:\#[0-9]+]]
// NOINLINE: ret void
// Swap the values of Buf at indices ix1 and ix2
void swap(unsigned Buf[MAX], unsigned ix1, unsigned ix2) {
  float tmp = Buf[ix1];
  Buf[ix1] = Buf[ix2];
  Buf[ix2] = tmp;
}

// NOINLINE: Function Attrs: alwaysinline
// NOINLINE: define void @_Z10BubbleSortA100_jj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %size) [[IntAttr]]
// NOINLINE: ret void
// Inefficiently sort Buf in place
void BubbleSort(unsigned Buf[MAX], unsigned size) {
  bool swapped = true;
  while (swapped) {
    swapped = false;
    for (unsigned i = 1; i < size; i++) {
      if (Buf[i] < Buf[i-1]) {
	swap(Buf, i, i-1);
	swapped = true;
      }
    }
  }
}

// Note ExtAttr is the inlined export set of attribs
// CHECK: Function Attrs: alwaysinline
// CHECK: define noundef i32 @_Z11RemoveDupesA100_jj(ptr {{[a-z_ ]*}}noundef byval([100 x i32]) align 4 {{.*}}%Buf, i32 noundef %size) {{[a-z_ ]*}}[[ExtAttr:\#[0-9]+]]
// CHECK: ret i32
// Sort Buf and remove any duplicate values
// returns the number of values left
export
unsigned RemoveDupes(unsigned Buf[MAX], unsigned size) {
  BubbleSort(Buf, size);
  unsigned insertPt = 0;
  for (unsigned i = 1; i < size; i++) {
    if (Buf[i] == Buf[i-1])
      insertPt++;
    else
      Buf[insertPt] = Buf[i];
  }
  return insertPt;
}


RWBuffer<unsigned> Indices;

// The mangled version of main only remains without inlining
// because it has internal linkage from the start
// Note main functions get the norecurse attrib, which IntAttr reflects
// NOINLINE: Function Attrs: alwaysinline
// NOINLINE: define internal void @_Z4mainj(i32 noundef %GI) [[IntAttr]]
// NOINLINE: ret void

// The unmangled version is not inlined, EntryAttr reflects that
// CHECK: Function Attrs: {{.*}}noinline
// CHECK: define void @main() {{[a-z_ ]*}}[[EntryAttr:\#[0-9]+]]
// Make sure function calls are inlined when AlwaysInline is run
// This only leaves calls to llvm. intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void

[numthreads(1,1,1)]
[shader("compute")]
void main(unsigned int GI : SV_GroupIndex) {
  unsigned tmpIndices[MAX];
  if (GI > MAX) return;
  for (unsigned i = 1; i < GI; i++)
    tmpIndices[i] = Indices[i];
  RemoveDupes(tmpIndices, GI);
  for (unsigned i = 1; i < GI; i++)
    tmpIndices[i] = Indices[i];
}

// The mangled version of main only remains without inlining
// because it has internal linkage from the start
// Note main functions get the norecurse attrib, which IntAttr reflects
// NOINLINE: Function Attrs: alwaysinline
// NOINLINE: define internal void @_Z6main10v() [[IntAttr]]
// NOINLINE: ret void

// The unmangled version is not inlined, EntryAttr reflects that
// CHECK: Function Attrs: {{.*}}noinline
// CHECK: define void @main10() {{[a-z_ ]*}}[[EntryAttr]]
// Make sure function calls are inlined when AlwaysInline is run
// This only leaves calls to llvm. intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void

[numthreads(1,1,1)]
[shader("compute")]
void main10() {
  main(10);
}

// NOINLINE: attributes [[IntAttr]] = {{.*}} alwaysinline
// CHECK: attributes [[ExtAttr]] = {{.*}} alwaysinline
// CHECK: attributes [[EntryAttr]] = {{.*}} noinline
