// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -debug-info-kind=limited -fblocks -emit-llvm -o - %s | FileCheck %s
// Assignment and block entry should point to the same line.
// rdar://problem/14039866

// CHECK: define{{.*}}@main()
// CHECK: store{{.*}}bitcast{{.*}}, !dbg ![[ASSIGNMENT:[0-9]+]]
// CHECK: define {{.*}} @__main_block_invoke
// CHECK: , !dbg ![[BLOCK_ENTRY:[0-9]+]]

int main(void)
{
// CHECK: [[ASSIGNMENT]] = !DILocation(line: [[@LINE+2]],
// CHECK: [[BLOCK_ENTRY]] = !DILocation(line: [[@LINE+1]],
    int (^blockptr)(void) = ^(void) {
      return 0;
    };
    return blockptr();
}

