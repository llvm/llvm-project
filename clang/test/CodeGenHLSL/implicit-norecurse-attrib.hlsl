// RUN: %clang_cc1 -x hlsl -triple dxil-pc-shadermodel6.3-library  -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -x hlsl -triple dxil-pc-shadermodel6.0-compute  -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Verify that a few different function types all get the NoRecurse attribute

#define MAX 100

struct Node {
  uint value;
  uint key;
  uint left, right;
};

// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define noundef i32 @_Z4FindA100_4Nodej(ptr noundef byval([100 x %struct.Node]) align 4 %SortedTree, i32 noundef %key) [[IntAttr:\#[0-9]+]]
// CHECK: ret i32
// Find and return value corresponding to key in the SortedTree
uint Find(Node SortedTree[MAX], uint key) {
  uint nix = 0; // head
  while(true) {
    if (nix < 0)
      return 0.0; // Not found
    Node n = SortedTree[nix];
    if (n.key == key)
      return n.value;
    if (key < n.key)
      nix = n.left;
    else
      nix = n.right;
  }
}

// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define noundef i1 @_Z8InitTreeA100_4NodeN4hlsl8RWBufferIDv4_jEEj(ptr noundef byval([100 x %struct.Node]) align 4 %tree, ptr noundef byval(%"class.hlsl::RWBuffer") align 4 %encodedTree, i32 noundef %maxDepth) [[ExtAttr:\#[0-9]+]]
// CHECK: ret i1
// Initialize tree with given buffer
// Imagine the inout works
export
bool InitTree(/*inout*/ Node tree[MAX], RWBuffer<uint4> encodedTree, uint maxDepth) {
  uint size = pow(2.f, maxDepth) - 1;
  if (size > MAX) return false;
  for (uint i = 1; i < size; i++) {
    tree[i].value = encodedTree[i].x;
    tree[i].key   = encodedTree[i].y;
    tree[i].left  = encodedTree[i].z;
    tree[i].right = encodedTree[i].w;
  }
  return true;
}

RWBuffer<uint4> gTree;

// Mangled entry points are internal
// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define internal void @_Z4mainj(i32 noundef %GI) [[IntAttr]]
// CHECK: ret void

// Canonical entry points are external and shader attributed
// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define void @main() [[EntryAttr:\#[0-9]+]]
// CHECK: ret void

[numthreads(1,1,1)]
[shader("compute")]
void main(uint GI : SV_GroupIndex) {
  Node haystack[MAX];
  uint needle = 0;
  if (InitTree(haystack, gTree, GI))
    needle = Find(haystack, needle);
}

// Mangled entry points are internal
// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define internal void @_Z11defaultMainv() [[IntAttr]]
// CHECK: ret void

// Canonical entry points are external and shader attributed
// CHECK: Function Attrs:{{.*}}norecurse
// CHECK: define void @defaultMain() [[EntryAttr]]
// CHECK: ret void

[numthreads(1,1,1)]
[shader("compute")]
void defaultMain() {
  Node haystack[MAX];
  uint needle = 0;
  if (InitTree(haystack, gTree, 4))
    needle = Find(haystack, needle);
}

// CHECK: attributes [[IntAttr]] = {{.*}} norecurse
// CHECK: attributes [[ExtAttr]] = {{.*}} norecurse
// CHECK: attributes [[EntryAttr]] = {{.*}} norecurse
