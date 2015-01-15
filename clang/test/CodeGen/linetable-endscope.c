// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Check the line numbers for the ret instruction. We expect it to be
// at the closing of the lexical scope in this case. See the comments in
// CodeGenFunction::FinishFunction() for more details.

// CHECK: define {{.*}}foo
// CHECK: store {{.*}}, !dbg ![[CONV:[0-9]+]]
// CHECK: ret {{.*}}, !dbg ![[RET:[0-9]+]]

void foo(char c)
{
  int i;
  // CHECK: ![[CONV]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  i = c;
  // CHECK: ![[RET]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}
