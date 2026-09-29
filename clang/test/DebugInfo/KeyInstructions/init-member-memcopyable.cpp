// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -gno-column-info -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s

// g::h can be memcpy'd (in this case emitted as load/stored), check the
// assignment gets Key Instructions metadata.

struct e {
  e(e&);
  e& operator=(const e&);
};

struct g {
  e f;
  int h;
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
// CHECK-NEXT: %2 = load ptr, ptr %.addr, align 8
// CHECK-NEXT: %h = getelementptr inbounds nuw %struct.g, ptr %2, i32 0, i32 1
// CHECK-NEXT: %3 = load i32, ptr %h, align 4, !dbg [[S1_G1R2:!.*]]
// CHECK-NEXT: %h2 = getelementptr inbounds nuw %struct.g, ptr %this1, i32 0, i32 1
// CHECK-NEXT: store i32 %3, ptr %h2, align 4, !dbg [[S1_G1R1:!.*]]
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
// CHECK-NEXT: %3 = load i32, ptr %h2, align 4, !dbg [[S2_G1R2:!.*]]
// CHECK-NEXT: store i32 %3, ptr %h, align 4, !dbg [[S2_G1R1:!.*]]
// CHECK-NEXT: ret void, !dbg

// CHECK: [[S1:!.*]] = distinct !DISubprogram(name: "operator=",
// CHECK: [[S1_G1R2]] = !DILocation(line: 12, scope: [[S1]], atomGroup: 1, atomRank: 2)
// CHECK: [[S1_G1R1]] = !DILocation(line: 12, scope: [[S1]], atomGroup: 1, atomRank: 1)

// CHECK: [[S2:!.*]] = distinct !DISubprogram(name: "g",
// CHECK: [[S2_G1R2]] = !DILocation(line: 12, scope: [[S2]], atomGroup: 1, atomRank: 2)
// CHECK: [[S2_G1R1]] = !DILocation(line: 12, scope: [[S2]], atomGroup: 1, atomRank: 1)

void fun(g *x) {
  g y = g(*x);
  y = *x;
}
