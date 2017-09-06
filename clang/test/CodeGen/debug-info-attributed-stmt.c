// RUN: %clang_cc1 -triple x86_64-unk-unk -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

void f(_Bool b)
{
#pragma nounroll
  while (b);
}

// CHECK: br label %while.cond, !dbg ![[NUM:[0-9]+]]
// CHECK: br i1 %tobool, label %while.body, label %while.end, !dbg ![[NUM]]
// CHECK: br label %while.cond, !dbg ![[NUM]], !llvm.loop
// CHECK: ![[NUM]] = !DILocation(line: 6,
