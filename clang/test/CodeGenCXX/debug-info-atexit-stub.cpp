// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-windows-msvc -gcodeview -debug-info-kind=limited -o - | FileCheck %s

struct a {
  ~a();
};
template <typename b> struct c : a {
  c(void (b::*)());
};
struct B {
  virtual void e();
};
c<B> *d() {
  static c<B> f(&B::e);
  return &f;
}

// CHECK: define internal void @"??__Ff@?1??d@@YAPEAU?$c@UB@@@@XZ@YAXXZ"()
// CHECK-SAME: !dbg ![[SUBPROGRAM:[0-9]+]] {
// CHECK: call void @"??1?$c@UB@@@@QEAA@XZ"(ptr @"?f@?1??d@@YAPEAU?$c@UB@@@@XZ@4U2@A"), !dbg ![[LOCATION:[0-9]+]]
// CHECK: ![[SUBPROGRAM]] = distinct !DISubprogram(name: "`dynamic atexit destructor for 'f'"
// CHECK-SAME: flags: DIFlagArtificial
// CHECK: ![[LOCATION]] = !DILocation(line: 0, scope: ![[SUBPROGRAM]])
