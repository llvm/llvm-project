// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - \
// RUN: | FileCheck %s

void *memset(void *, int, unsigned long);
void a(int *P) {
    memset(P, 1, 8);
}

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
