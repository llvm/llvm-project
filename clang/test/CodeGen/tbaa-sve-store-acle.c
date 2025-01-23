// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -O1  %s \
// RUN:     -emit-llvm -o - | FileCheck %s -check-prefix=TBAA
#include <arm_sve.h>

// TBAA:    store <vscale x 2 x i64>
// TBAA:    !tbaa ![[TBAA6:[0-9]+]]
long long sveStoreWithTypeCast(int *datas) {
  long long res2[16];
  svbool_t pa = svptrue_b32();
  svint32_t v1 = svld1(pa, &datas[0]);
  svint64_t v2 = svunpklo(v1);
  svst1(pa, (long *)&res2[0], v2);
  return res2[0] + res2[1];
}

// TBAA: ![[CHAR:[0-9]+]] = !{!"omnipotent char",
// TBAA: ![[TBAA6:[0-9]+]] = !{![[CHAR]], ![[CHAR]], i64 0}
