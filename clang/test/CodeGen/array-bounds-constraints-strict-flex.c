// Verify that -fstrict-flex-arrays controls whether a trailing array is treated
// as a flexible array member for the purpose of the array bounds assumes
// emitted by -fassume-array-bounds. Use -O1 -disable-llvm-optzns to check the
// assumes before DropUnnecessaryAssumesPass drops them.
// RUN: %clang_cc1 -emit-llvm -O1 -disable-llvm-optzns -fassume-array-bounds -fstrict-flex-arrays=0 %s -o - | FileCheck %s --check-prefixes=CHECK,LEVEL0
// RUN: %clang_cc1 -emit-llvm -O1 -disable-llvm-optzns -fassume-array-bounds -fstrict-flex-arrays=3 %s -o - | FileCheck %s --check-prefixes=CHECK,LEVEL3

struct TrailingOne {
  int count;
  int data[1];
};

// CHECK-LABEL: define {{.*}} @access_trailing_size_one
int access_trailing_size_one(struct TrailingOne *s, int i) {
  // At -fstrict-flex-arrays=0 the trailing "data[1]" is treated as a flexible
  // array member, so no bounds assume is emitted.
  // LEVEL0-NOT: call void @llvm.assume
  // At -fstrict-flex-arrays=3 only "data[]" is flexible, so "data[1]" is a real
  // one-element array and a bounds assume is emitted.
  // LEVEL3: call void @llvm.assume{{.*}}!llvm.array.bounds
  return s->data[i];
}

struct TrailingIncomplete {
  int count;
  int data[];
};

// CHECK-LABEL: define {{.*}} @access_trailing_incomplete
int access_trailing_incomplete(struct TrailingIncomplete *s, int i) {
  // A true flexible array member ("data[]") is never given a bounds assume,
  // regardless of the -fstrict-flex-arrays level.
  // CHECK-NOT: call void @llvm.assume
  return s->data[i];
}
