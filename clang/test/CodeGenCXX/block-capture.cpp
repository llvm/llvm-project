// RUN: %clang_cc1 -x c++ -std=c++11 -fblocks -emit-llvm %s -o - | FileCheck %s

// CHECK: %struct.__block_byref_baz = type { ptr, ptr, i32, i32, i32 }
// CHECK: [[baz:%[0-9a-z_]*]] = alloca %struct.__block_byref_baz
// CHECK: [[bazref:%[0-9a-z_\.]*]] = getelementptr inbounds nuw %struct.__block_byref_baz, ptr [[baz]], i32 0, i32 1
// CHECK: store ptr [[baz]], ptr [[bazref]]
// CHECK: call void @_Block_object_dispose(ptr [[baz]]

int main() {
  __block int baz = [&]() { return 0; }();
  ^{ (void)baz; };
  return 0;
}
