// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -disable-llvm-passes -emit-llvm -o %t2.ll %s | FileCheck %s --check-prefixes=MM,MM2
// RUN: FileCheck %s --check-prefixes=LL,LL2 < %t2.ll

// LL: define{{.+}}func_cond{{.+}}(
// MM: func_cond{{.*}}:
int func_cond(bool a, bool b) {
  // %mcdc.addr* are emitted by static order.
  // LL:   %[[MA4:mcdc.addr.*]] = alloca i32, align 4
  // LL:   %[[MA6:mcdc.addr.*]] = alloca i32, align 4
  // LL:   %[[MA7:mcdc.addr.*]] = alloca i32, align 4
  // LL:   %[[MA9:mcdc.addr.*]] = alloca i32, align 4
  // LL:   %[[MA10:mcdc.addr.*]] = alloca i32, align 4
  // LL:  call void @llvm.instrprof.mcdc.parameters(ptr @[[PROFN:.+]], i64 [[#H:]], i32 [[#BS:]])
  int count = 0;
  if (a)
    // NB=2 Single cond
    // MM2-NOT: Decision
    ++count;
  if (a ? true : false)
    // NB=2,2 Wider decision comes first.
    // MA2 has C:2
    // MA3 has C:1
    ++count;
  if (a && b ? true : false)
    // NB=2,3 Wider decision comes first.
    // MM2:  Decision,File 0, [[@LINE-2]]:7 -> [[#L:@LINE-2]]:13 = M:[[#I:3]], C:2
    // MM:   Branch,File 0, [[#L]]:7 -> [[#L]]:8 = #6, (#0 - #6) [1,2,0]
    // MM:   Branch,File 0, [[#L]]:12 -> [[#L]]:13 = #7, (#6 - #7) [2,0,0]
    // LL:   store i32 0, ptr %[[MA4]], align 4
    // LL:   = load i32, ptr %[[MA4]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA4]], align 4
    // LL:   = load i32, ptr %[[MA4]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA4]], align 4
    // LL2:  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @[[PROFN]], i64 [[#H]], i32 [[#B:0]], ptr %[[MA4]])
    ++count;
  while (a || true) {
    // NB=3 BinOp only
    // MM:   Decision,File 0, [[@LINE-2]]:10 -> [[#L:@LINE-2]]:19 = M:[[#I:I+3]], C:2
    // MM:   Branch,File 0, [[#L]]:10 -> [[#L]]:11 = (#0 - #9), #9 [1,0,2]
    // MM:   Branch,File 0, [[#L]]:15 -> [[#L]]:19 = (#9 - #10), 0 [2,0,0]
    // LL:   store i32 0, ptr %[[MA6]], align 4
    // LL:   = load i32, ptr %[[MA6]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA6]], align 4
    // LL:   = load i32, ptr %[[MA6]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA6]], align 4
    // LL2:  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @[[PROFN]], i64 [[#H]], i32 [[#B:B+3]], ptr %[[MA6]])
    ++count;
    break;
  }
  while (a || true ? false : true) {
    // Wider decision comes first.
    // MM2:  Decision,File 0, [[@LINE-2]]:10 -> [[#L:@LINE-2]]:19 = M:[[#I:I+3]], C:2
    // MM:   Branch,File 0, [[#L]]:10 -> [[#L]]:11 = ((#0 + #11) - #13), #13 [1,0,2]
    // MM:   Branch,File 0, [[#L]]:15 -> [[#L]]:19 = (#13 - #14), 0 [2,0,0]
    // LL:   store i32 0, ptr %[[MA7]], align 4
    // LL:   = load i32, ptr %[[MA7]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA7]], align 4
    // LL:   = load i32, ptr %[[MA7]], align 4
    // LL:   store i32 %{{.+}}, ptr %[[MA7]], align 4
    // LL:   call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @[[PROFN]], i64 [[#H]], i32 [[#B:B+3]], ptr %[[MA7]])
    ++count;
  }
  do {
    ++count;
  } while (a && false);
  // BinOp only
  // MM:   Decision,File 0, [[@LINE-2]]:12 -> [[#L:@LINE-2]]:22 = M:[[#I:I+3]], C:2
  // MM:   Branch,File 0, [[#L]]:12 -> [[#L]]:13 = #16, ((#0 + #15) - #16) [1,2,0]
  // MM:   Branch,File 0, [[#L]]:17 -> [[#L]]:22 = 0, (#16 - #17) [2,0,0]
  // LL:   store i32 0, ptr %[[MA9]], align 4
  // LL:   = load i32, ptr %[[MA9]], align 4
  // LL:   store i32 %{{.+}}, ptr %[[MA9]], align 4
  // LL:   = load i32, ptr %[[MA9]], align 4
  // LL:   store i32 %{{.+}}, ptr %[[MA9]], align 4
  // LL2:  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @[[PROFN]], i64 [[#H]], i32 [[#B:B+3]], ptr %[[MA9]])
  do {
    ++count;
  } while (a && false ? true : false);
  // Wider decision comes first.
  // MM2:  Decision,File 0, [[@LINE-2]]:12 -> [[#L:@LINE-2]]:22 = M:15, C:2
  // MM:   Branch,File 0, [[#L]]:12 -> [[#L]]:13 = #20, ((#0 + #18) - #20) [1,2,0]
  // MM:   Branch,File 0, [[#L]]:17 -> [[#L]]:22 = 0, (#20 - #21) [2,0,0]
  // LL:   store i32 0, ptr %[[MA10]], align 4
  // LL:   = load i32, ptr %[[MA10]], align 4
  // LL:   store i32 %{{.+}}, ptr %[[MA10]], align 4
  // LL:   = load i32, ptr %[[MA10]], align 4
  // LL:   store i32 %{{.+}}, ptr %[[MA10]], align 4
  // LL:   call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @[[PROFN]], i64 [[#H]], i32 [[#B:B+3]], ptr %[[MA10]])
  // FIXME: Confirm (B+3==BS)
  for (int i = 0; i < (a ? 2 : 1); ++i) {
    // Simple nested decision (different column)
    // MM2-NOT: Decision
    // LL2-NOT: call void @llvm.instrprof.mcdc.tvbitmap.update
    ++count;
  }
  for (int i = 0; i >= 4 ? false : true; ++i) {
    // Wider decision comes first.
    ++count;
  }
  return count;
}
