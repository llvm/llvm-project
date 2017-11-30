// RUN: %clang_cc1 %s -triple hexagon-unknown-elf -O2 -emit-llvm -o - | FileCheck %s

typedef union __attribute__((aligned(4))) {
  unsigned short uh[2];
  unsigned uw;
} vect32;

void bar(vect32 p[][2]);

// CHECK-LABEL: define void @fred
void fred(unsigned Num, int Vec[2], int *Index, int Arr[4][2]) {
  vect32 Tmp[4][2];
// Generate tbaa for the load of Index:
// CHECK: load i32, i32* %Index{{.*}}tbaa
// But no tbaa for the two stores:
// CHECK: %uw[[UW1:[0-9]*]] = getelementptr
// CHECK: store{{.*}}%uw[[UW1]]
// CHECK: tbaa [[TAG_vect32_union_member:![0-9]+]]
// There will be a load after the store, and it will use tbaa. Make sure
// the check-not above doesn't find it:
// CHECK: load
  Tmp[*Index][0].uw = Arr[*Index][0] * Num;
// CHECK: %uw[[UW2:[0-9]*]] = getelementptr
// CHECK: store{{.*}}%uw[[UW2]]
// CHECK: tbaa [[TAG_vect32_union_member]]
  Tmp[*Index][1].uw = Arr[*Index][1] * Num;
// Same here, don't generate tbaa for the loads:
// CHECK: %uh[[UH1:[0-9]*]] = bitcast %union.vect32
// CHECK: %arrayidx[[AX1:[0-9]*]] = getelementptr{{.*}}%uh[[UH1]]
// CHECK: load i16, i16* %arrayidx[[AX1]]
// CHECK: tbaa [[TAG_vect32_union_member]]
// CHECK: store
  Vec[0] = Tmp[*Index][0].uh[1];
// CHECK: %uh[[UH2:[0-9]*]] = bitcast %union.vect32
// CHECK: %arrayidx[[AX2:[0-9]*]] = getelementptr{{.*}}%uh[[UH2]]
// CHECK: load i16, i16* %arrayidx[[AX2]]
// CHECK: tbaa [[TAG_vect32_union_member]]
// CHECK: store
  Vec[1] = Tmp[*Index][1].uh[1];
  bar(Tmp);
}

// CHECK-DAG: [[TAG_vect32_union_member]] = !{[[TYPE_vect32:!.*]], [[TYPE_union_member:!.*]], i64 0}
// CHECK-DAG: [[TYPE_vect32]] = !{!"", [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TYPE_union_member]] = !{!"union member", [[TYPE_char:!.*]], i64 0}
// CHECK-DAG: [[TYPE_char]] = !{!"omnipotent char", {{.*}}}
