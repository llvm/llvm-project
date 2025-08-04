// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -gno-column-info -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s

// g::h and i can be memcpy'd, check the assignment gets Key Instructions metadata.

struct e {
  e(e &);
  e& operator=(const e&);
};

struct g {
  e f;
  int h;
  int i;
};

// Copy assignment operator.
// CHECK: define{{.*}}ptr @_ZN1gaSERKS_
// CHECK-NEXT: entry:
// CHECK-NEXT: %this.addr = alloca ptr, align 8
// CHECK-NEXT: %.addr = alloca ptr, align 8
// CHECK-NEXT: store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT: store ptr %0, ptr %.addr, align 8
// CHECK-NEXT: %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT: %1 = load ptr, ptr %.addr, align 8
// CHECK-NEXT: %call = call {{.*}}ptr @_ZN1eaSERKS_(ptr {{.*}}%this1, ptr {{.*}}%1)
// CHECK-NEXT: %h = getelementptr inbounds nuw %struct.g, ptr %this1, i32 0, i32 1
// CHECK-NEXT: %2 = load ptr, ptr %.addr, align 8
// CHECK-NEXT: %h2 = getelementptr inbounds nuw %struct.g, ptr %2, i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy{{.*}}(ptr align 4 %h, ptr align 4 %h2, i64 8, i1 false), !dbg [[S1_G1R1:!.*]]
// CHECK-NEXT: ret ptr %this1, !dbg

// Copy ctor.
// CHECK: define{{.*}}void @_ZN1gC2ERS_
// CHECK-NEXT: entry:
// CHECK-NEXT: %this.addr = alloca ptr, align 8
// CHECK-NEXT: %.addr = alloca ptr, align 8
// CHECK-NEXT: store ptr %this, ptr %this.addr, align 8
// CHECK-NEXT: store ptr %0, ptr %.addr, align 8
// CHECK-NEXT: %this1 = load ptr, ptr %this.addr, align 8
// CHECK-NEXT: %1 = load ptr, ptr %.addr, align 8
// CHECK-NEXT: call void @_ZN1eC1ERS_
// CHECK-NEXT: %h = getelementptr inbounds nuw %struct.g, ptr %this1, i32 0, i32 1
// CHECK-NEXT: %2 = load ptr, ptr %.addr, align 8
// CHECK-NEXT: %h2 = getelementptr inbounds nuw %struct.g, ptr %2, i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy{{.*}}(ptr align 4 %h, ptr align 4 %h2, i64 8, i1 false), !dbg [[S2_G1R1:!.*]]
// CHECK-NEXT: ret void, !dbg

// CHECK: [[S1:!.*]] = distinct !DISubprogram(name: "operator=",
// CHECK: [[S1_G1R1]] = !DILocation(line: 11, scope: [[S1]], atomGroup: 1, atomRank: 1)

// CHECK: [[S2:!.*]] = distinct !DISubprogram(name: "g",
// CHECK: [[S2_G1R1]] = !DILocation(line: 11, scope: [[S2]], atomGroup: 1, atomRank: 1)

[[gnu::nodebug]]
void fun(g *x) {
  g y = g(*x);
  y = *x;
}
