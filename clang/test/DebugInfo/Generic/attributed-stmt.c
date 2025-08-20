// RUN: %clang_cc1 -triple x86_64-unk-unk -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

void f(_Bool b)
{
#pragma nounroll
  while (b);
}

// CHECK: br label {{.*}}, !dbg ![[NUM:[0-9]+]]
// CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !dbg ![[NUM]]
// CHECK: br label {{.*}}, !dbg ![[NUM]], !llvm.loop
// CHECK: ![[NUM]] = !DILocation(line: 6,
