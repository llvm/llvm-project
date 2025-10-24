// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o - -fexceptions -fcxx-exceptions \
// RUN: | FileCheck %s

void except() {
  // FIXME(OCH): Should `store i32 32, ptr %exception` be key?
  throw 32;
}

void attempt() {
  try { except(); }
// CHECK: catch:
// CHECK: %4 = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK: %5 = load i32{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store i32 %5, ptr %e{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: call void @__cxa_end_catch()
  catch (int e) { }

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
