// RUN: %clang_cc1 -no-opaque-pointers -x c++ -std=c++11 -fblocks -emit-llvm %s -o - | FileCheck %s

// CHECK: %struct.__block_byref_baz = type { i8*, %struct.__block_byref_baz*, i32, i32, i32 }
// CHECK: [[baz:%[0-9a-z_]*]] = alloca %struct.__block_byref_baz
// CHECK: [[bazref:%[0-9a-z_\.]*]] = getelementptr inbounds %struct.__block_byref_baz, %struct.__block_byref_baz* [[baz]], i32 0, i32 1
// CHECK: store %struct.__block_byref_baz* [[baz]], %struct.__block_byref_baz** [[bazref]]
// CHECK: bitcast %struct.__block_byref_baz* [[baz]] to i8*
// CHECK: [[disposable:%[0-9a-z_]*]] = bitcast %struct.__block_byref_baz* [[baz]] to i8*
// CHECK: call void @_Block_object_dispose(i8* [[disposable]]

int main() {
  __block int baz = [&]() { return 0; }();
  ^{ (void)baz; };
  return 0;
}
