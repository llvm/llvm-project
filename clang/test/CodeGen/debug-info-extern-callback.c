// RUN: %clang_cc1 -x c -debug-info-kind=limited -triple bpf-linux-gnu -emit-llvm %s -o - | FileCheck %s

extern int do_work1(int);
long bpf_helper1(void *callback_fn);
long prog1(void) {
  return bpf_helper1(&do_work1);
}

extern int do_work2(void);
long prog2_1(void) {
  return (long)&do_work2;
}
int do_work2(void) { return 0; }
long prog2_2(void) {
  return (long)&do_work2;
}

// CHECK: declare !dbg ![[FUNC1:[0-9]+]] i32 @do_work1
// CHECK: define dso_local i32 @do_work2() #{{[0-9]+}} !dbg ![[FUNC2:[0-9]+]]

// CHECK: ![[FUNC1]] = !DISubprogram(name: "do_work1"
// CHECK: ![[FUNC2]] = distinct !DISubprogram(name: "do_work2"
