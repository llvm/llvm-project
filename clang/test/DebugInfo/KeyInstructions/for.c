// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Perennial question: should the inc be its own source atom or not
// (currently it is).

// TODO: See do.c and while.c regarding cmp and cond br groups.

void a(int A) {
// CHECK: entry:
// CHECK: store i32 0, ptr %i{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: for.cond:
// CHECK: %cmp = icmp slt i32 %0, %1, !dbg [[G2R1:!.*]]
// CHECK: br i1 %cmp, label %for.body, label %for.end, !dbg [[G3R1:!.*]]

// TODO: The unconditional br is given an atom group here which is useful for
// O0. Since we're no longer targeting O0 we should reevaluate whether this
// adds any value.
// CHECK: for.body:
// CHECK-NEXT: br label %for.inc, !dbg [[G5R1:!.*]]

// CHECK: for.inc:
// CHECK: %inc = add{{.*}}, !dbg [[G4R2:!.*]]
// CHECK: store i32 %inc, ptr %i{{.*}}, !dbg [[G4R1:!.*]]
  for (int i = 0; i < A; ++i) {
  }
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

void b(int A) {
// CHECK: entry:
// CHECK: store i32 0, ptr %i{{.*}}, !dbg [[bG1R1:!.*]]
// CHECK: for.cond:
// CHECK: %cmp = icmp slt i32 %0, %1, !dbg [[bG2R1:!.*]]
// CHECK: br i1 %cmp, label %for.body, label %for.end, !dbg [[bG3R1:!.*]]

// CHECK: for.body:
// CHECK-NEXT: %2 = load i32, ptr %A.addr
// - If stmt atom:
// CHECK-NEXT: %cmp1 = icmp sgt i32 %2, 1, !dbg [[bG4R2:!.*]]
// CHECK-NEXT: br i1 %cmp1, label %if.then, label %if.end, !dbg [[bG4R1:!.*]]
// CHECK: if.then:
// CHECK-NEXT: br label %if.end

// - For closing brace.
// CHECK: if.end:
// CHECK-NEXT: br label %for.inc, !dbg [[bG6R1:!.*]]

// CHECK: for.inc:
// CHECK: %inc = add{{.*}}, !dbg [[bG5R2:!.*]]
// CHECK: store i32 %inc, ptr %i{{.*}}, !dbg [[bG5R1:!.*]]
  for (int i = 0; i < A; ++i) {
    if (A > 1)
      ;
  }
// CHECK: ret{{.*}}, !dbg [[bRET:!.*]]
}

void c(int A) {
// CHECK: entry:
// CHECK: for.cond:
// CHECK-NEXT: %0 = load i32, ptr %A.addr
// - If stmt atom:
// CHECK-NEXT: %cmp = icmp sgt i32 %0, 1, !dbg [[cG1R2:!.*]]
// CHECK-NEXT: br i1 %cmp, label %if.then, label %if.end, !dbg [[cG1R1:!.*]]
// CHECK: if.then:
// CHECK-NEXT: br label %if.end

// - For closing brace.
// CHECK: if.end:
// CHECK-NEXT: br label %for.inc, !dbg [[cG3R1:!.*]]

// CHECK: for.inc:
// CHECK-NEXT: %1 = load i32, ptr %A.addr
// CHECK-NEXT: %inc = add{{.*}}, !dbg [[cG2R2:!.*]]
// CHECK-NEXT: store i32 %inc, ptr %A.addr{{.*}}, !dbg [[cG2R1:!.*]]
  for (; /*no cond*/ ; ++A) {
    if (A > 1)
      ;
  }
}

void d() {
// - Check the `for` keyword gets an atom group.
// CHECK: entry:
// CHECK-NEXT: br label %for.cond

// CHECK: for.cond:
// CHECK: br label %for.cond, !dbg [[dG1R1:!.*]], !llvm.loop
  for ( ; ; )
  {
  }
}

int x, i;
void ee();
void e() {
// - Check we assign an atom group to `for.body`s `br`, even without braces.
// - TODO: Investigate whether this is needed.
// CHECK: entry:

// CHECK: for.cond:
// CHECK-NEXT: %0 = load i32, ptr @i
// CHECK-NEXT: %cmp = icmp slt i32 %0, 3, !dbg [[eG1R1:!.*]]
// CHECK-NEXT: br i1 %cmp, label %for.body, label %for.end, !dbg [[eG2R1:!.*]]

// CHECK: for.body:
// CHECK-NEXT: %1 = load i32, ptr @i{{.*}}, !dbg [[eG3R2:!.*]]
// CHECK-NEXT: store i32 %1, ptr @x{{.*}}, !dbg [[eG3R1:!.*]]
// CHECK-NEXT: br label %for.inc, !dbg [[eG4R1:!.*]]
  for (; i < 3; ee())
    x = i;
// CHECK: ret{{.*}}, !dbg [[eRET:!.*]]
}


void f() {
// - Check the `continue` keyword gets an atom group.
// CHECK: entry:
// CHECK-NEXT: br label %for.cond

// CHECK: for.cond:
// CHECK: br label %for.cond, !dbg [[fG1R1:!.*]], !llvm.loop
  for ( ; ; )
  {
    continue;
  }
}

void g() {
// - Check the `break` keyword gets an atom group.
// CHECK: entry:
// CHECK-NEXT: br label %for.cond

// CHECK: for.cond:
// CHECK: br label %for.end, !dbg [[gG1R1:!.*]]
  for ( ; ; )
  {
    break;
  }
// CHECK: ret{{.*}}, !dbg [[gRET:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G5R1]] = !DILocation(line: 29,{{.*}} atomGroup: 5, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)

// CHECK: [[bG1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[bG2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[bG3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[bG4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[bG4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[bG6R1]] = !DILocation(line: 58,{{.*}} atomGroup: 6, atomRank: 1)
// CHECK: [[bG5R2]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 2)
// CHECK: [[bG5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[bRET]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)

// CHECK: [[cG1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[cG1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[cG3R1]] = !DILocation(line: 83,{{.*}} atomGroup: 3, atomRank: 1)
// CHECK: [[cG2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[cG2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)

// CHECK: [[dG1R1]] = !DILocation(line: 93, column: 3, scope: ![[#]], atomGroup: 1, atomRank: 1)

// CHECK: [[eG1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[eG2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[eG3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[eG3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[eG4R1]] = !DILocation(line: 115, column: 5, scope: ![[#]], atomGroup: 4, atomRank: 1)
// CHECK: [[eRET]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)

// CHECK: [[fG1R1]] = !DILocation(line: 129, column: 5, scope: ![[#]], atomGroup: 1, atomRank: 1)

// CHECK: [[gG1R1]] = !DILocation(line: 142, column: 5, scope: ![[#]], atomGroup: 1, atomRank: 1)
// CHECK: [[gRET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
