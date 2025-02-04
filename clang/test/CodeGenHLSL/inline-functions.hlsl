// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK_LIB_OPTNONE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O0 -o - | FileCheck %s --check-prefixes=CHECK,CHECK_LIB_OPTNONE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,CHECK_OPT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK_CS_OPTNONE_NOPASS
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -O0 -o - | FileCheck %s --check-prefixes=CHECK,CHECK_CS_OPTNONE_PASS
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute %s -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK,CHECK_OPT

// Tests inlining of user functions based on specified optimization options.
// This includes exported functions and mangled entry point implementation functions.

#define MAX 100

float nums[MAX];

// Check optnone attribute for library target compilation
// CHECK_LIB_OPTNONE: Function Attrs:{{.*}}optnone
// CHECK_LIB_OPTNONE: define void @_Z4swapA100_jjj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %ix1, i32 noundef %ix2) [[ExtAttr:\#[0-9]+]]

// Check alwaysinline attribute for non-entry functions of compute target compilation
// CHECK_CS_OPTNONE_NOPASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_NOPASS: define void @_Z4swapA100_jjj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %ix1, i32 noundef %ix2) [[ExtAttr:\#[0-9]+]]

// Check alwaysinline attribute for non-entry functions of compute target compilation
// CHECK_CS_OPTNONE_PASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_PASS: define void @_Z4swapA100_jjj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %ix1, i32 noundef %ix2) [[ExtAttr:\#[0-9]+]]

// Check alwaysinline attribute for opt compilation to library target
// CHECK_OPT: Function Attrs: alwaysinline
// CHECK_OPT: define void @_Z4swapA100_jjj(ptr noundef byval([100 x i32]) align 4 captures(none) %Buf, i32 noundef %ix1, i32 noundef %ix2) {{[a-z_ ]*}} [[SwapOptAttr:\#[0-9]+]]

// CHECK: ret void

// Swap the values of Buf at indices ix1 and ix2
void swap(unsigned Buf[MAX], unsigned ix1, unsigned ix2) {
  float tmp = Buf[ix1];
  Buf[ix1] = Buf[ix2];
  Buf[ix2] = tmp;
}

// Check optnone attribute for library target compilation
// CHECK_LIB_OPTNONE: Function Attrs:{{.*}}optnone
// CHECK_LIB_OPTNONE: define void @_Z10BubbleSortA100_jj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %size) [[ExtAttr]]

// Check alwaysinline attribute for non-entry functions of compute target compilation
// CHECK_CS_OPTNONE_NOPASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_NOPASS: define void @_Z10BubbleSortA100_jj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %size) [[ExtAttr]]

// Check alwaysinline attribute for non-entry functions of compute target compilation
// CHECK_CS_OPTNONE_PASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_PASS: define void @_Z10BubbleSortA100_jj(ptr noundef byval([100 x i32]) align 4 %Buf, i32 noundef %size) [[ExtAttr]]

// Check alwaysinline attribute for opt compilation to library target
// CHECK_OPT: Function Attrs: alwaysinline
// CHECK_OPT: define void @_Z10BubbleSortA100_jj(ptr noundef readonly byval([100 x i32]) align 4 captures(none) %Buf, i32 noundef %size) {{[a-z_ ]*}} [[BubOptAttr:\#[0-9]+]]

// CHECK: ret void

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

// Check optnone attribute for library target compilation of exported function
// CHECK_LIB_OPTNONE: Function Attrs:{{.*}}optnone
// CHECK_LIB_OPTNONE: define noundef i32 @_Z11RemoveDupesA100_jj(ptr {{[a-z_ ]*}}noundef byval([100 x i32]) align 4 {{.*}}%Buf, i32 noundef %size) [[ExportAttr:\#[0-9]+]]
// Sort Buf and remove any duplicate values
// returns the number of values left

// Check alwaysinline attribute for exported function of compute target compilation
// CHECK_CS_OPTNONE_NOPASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_NOPASS: define noundef i32 @_Z11RemoveDupesA100_jj(ptr {{[a-z_ ]*}}noundef byval([100 x i32]) align 4 {{.*}}%Buf, i32 noundef %size) [[ExportAttr:\#[0-9]+]]

// Check alwaysinline attribute for exported function of compute target compilation
// CHECK_CS_OPTNONE_PASS: Function Attrs: alwaysinline
// CHECK_CS_OPTNONE_PASS: define noundef i32 @_Z11RemoveDupesA100_jj(ptr {{[a-z_ ]*}}noundef byval([100 x i32]) align 4 {{.*}}%Buf, i32 noundef %size) [[ExportAttr:\#[0-9]+]]

// Check alwaysinline attribute for exported function of library target compilation
// CHECK_OPT: Function Attrs: alwaysinline
// CHECK_OPT: define noundef i32 @_Z11RemoveDupesA100_jj(ptr noundef byval([100 x i32]) align 4 captures(none) %Buf, i32 noundef %size) {{[a-z_ ]*}} [[RemOptAttr:\#[0-9]+]]

// CHECK: ret i32

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

// CHECK_LIB_OPTNONE: Function Attrs:{{.*}}optnone
// Internal function attributes are the same as those of source function's
// CHECK_LIB_OPTNONE: define internal void @_Z4mainj(i32 noundef %GI) [[ExtAttr]]
// CHECK_LIB_OPTNONE: ret void

// CHECK_CS_OPTNONE_NOPASS: Function Attrs: alwaysinline
// Internal function attributes are different from those of source function's
// CHECK_CS_OPTNONE_NOPASS: define internal void @_Z4mainj(i32 noundef %GI) [[ExtAttr]]
// CHECK_CS_OPTNONE_NOPASS: ret void

// Check internal function @_Z4mainj is not generated when LLVM passes enabled
// CHECK_CS_OPTNONE_PASS-NOT: define internal void @_Z4mainj

// Check internal function @_Z4mainj is not generated as it should be inlined
// for opt builds
// CHECK_OPT-NOT: define internal void @_Z4mainj

// The unmangled version is not inlined, EntryAttr reflects that
// CHECK_LIB_OPTNONE: Function Attrs: {{.*}}noinline
// CHECK_LIB_OPTNONE: define void @main() {{[a-z_ ]*}}[[EntryAttr:\#[0-9]+]]
// Make sure internal function is not inlined when optimization is disabled
// CHECK_LIB_OPTNONE: call void @_Z4mainj

// CHECK_CS_OPTNONE_NOPASS: Function Attrs:{{.*}}optnone
// CHECK_CS_OPTNONE_NOPASS: define void @main() {{[a-z_ ]*}}[[EntryAttr:\#[0-9]+]]
// Make sure internal function is not inlined when optimization is disabled
// CHECK_CS_OPTNONE_NOPASS: call void @_Z4mainj

// CHECK_CS_OPTNONE_PASS: Function Attrs: {{.*}}noinline
// CHECK_CS_OPTNONE_PASS: define void @main() {{[a-z_ ]*}}[[EntryAttr:\#[0-9]+]]
// Make sure internal function is inlined when LLVM passes are enabled
// CHECK_CS_OPTNONE_PASS: _Z4mainj.exit:

// CHECK_OPT: Function Attrs: {{.*}}noinline
// CHECK_OPT: define void @main() {{[a-z_ ]*}}[[EntryAttr:\#[0-9]+]]
// Make sure internal function is inlined as optimization is enabled
// CHECK_OPT: _Z4mainj.exit:

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

// CHECK_LIB_OPTNONE: Function Attrs:{{.*}}optnone
// CHECK_LIB_OPTNONE: define internal void @_Z6main10v() [[ExtAttr]]
// CHECK_LIB_OPTNONE: ret void

// CHECK_CS_OPTNONE_NOPASS: Function Attrs:{{.*}}alwaysinline
// CHECK_CS_OPTNONE_NOPASS: define internal void @_Z6main10v() [[ExtAttr]]
// CHECK_CS_OPTNONE_NOPASS: ret void

// Check internal function @_Z6main10v is not generated when LLVM passes are enabled
// CHECK_CS_OPTNONE_PASS-NOT: define internal void @_Z6main10v

// Check internal function @_Z6main10v is not generated as it should be inlined
// CHECK_OPT-NOT: define internal void @_Z6main10v

// The unmangled version is not inlined, EntryAttr reflects that
// CHECK_LIB_OPTNONE: Function Attrs: {{.*}}noinline
// CHECK_LIB_OPTNONE: define void @main10() {{[a-z_ ]*}}[[EntryAttr]]
// Make sure internal function is not inlined when optimization is disabled
// CHECK_LIB_OPTNONE: call void @_Z6main10v

// CHECK_CS_OPTNONE_NOPASS: Function Attrs: {{.*}}noinline
// CHECK_CS_OPTNONE_NOPASS: define void @main10() {{[a-z_ ]*}}[[EntryAttr]]
// Make sure internal function is not inlined when optimization is disabled
// CHECK_CS_OPTNONE_NOPASS: call void @_Z6main10v

// CHECK_CS_OPTNONE_PASS: Function Attrs: {{.*}}noinline
// CHECK_CS_OPTNONE_PASS: define void @main10() {{[a-z_ ]*}}[[EntryAttr]]
// Check internal function is inlined as optimization is enabled when LLVM passes
// are enabled
// CHECK_CS_OPTNONE_PASS: _Z6main10v.exit:

// CHECK_OPT: Function Attrs: {{.*}}noinline
// CHECK_OPT: define void @main10() {{[a-z_ ]*}}[[EntryAttr]]
// Make sure internal function is inlined as optimization is enabled
// CHECK_OPT: _Z6main10v.exit:
// CHECK: ret void

[numthreads(1,1,1)]
[shader("compute")]
void main10() {
  main(10);
}

// CHECK_LIB_OPTNONE: attributes [[ExtAttr]] = {{.*}} optnone
// CHECK_LIB_OPTNONE: attributes [[ExportAttr]] = {{.*}} optnone

// CHECK_CS_OPTNONE_NOPASS: attributes [[ExtAttr]] ={{.*}} alwaysinline
// CHECK_CS_OPTNONE_NOPASS: attributes [[EntryAttr]] = {{.*}} noinline

// CHECK_CS_OPTNONE_PASS: attributes [[ExtAttr]] ={{.*}} alwaysinline
// CHECK_CS_OPTNONE_PASS: attributes [[EntryAttr]] = {{.*}} noinline

// CHECK_OPT: attributes [[SwapOptAttr]] ={{.*}} alwaysinline
// CHECK_OPT: attributes [[BubOptAttr]] ={{.*}} alwaysinline
// CHECK_OPT: attributes [[RemOptAttr]] ={{.*}} alwaysinline
// CHECK_OPT: attributes [[EntryAttr]] ={{.*}} noinline
