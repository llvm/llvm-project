// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int s0(int a, int b) {
  int x = a + b;
  if (x > 0)
    x = 0;
  else
    x = 1;
  return x;
}

// CIR: #loc3 = loc("{{.*}}sourcelocation.cpp":6:8)
// CIR: #loc4 = loc("{{.*}}sourcelocation.cpp":6:12)
// CIR: #loc5 = loc("{{.*}}sourcelocation.cpp":6:15)
// CIR: #loc6 = loc("{{.*}}sourcelocation.cpp":6:19)
// CIR: #loc21 = loc(fused[#loc3, #loc4])
// CIR: #loc22 = loc(fused[#loc5, #loc6])
// CIR: module @"{{.*}}sourcelocation.cpp" attributes {cir.lang = #cir.lang<cxx>, cir.sob = #cir.signed_overflow_behavior<undefined>
// CIR:   cir.func @_Z2s0ii(%arg0: !s32i loc(fused[#loc3, #loc4]), %arg1: !s32i loc(fused[#loc5, #loc6])) -> !s32i
// CIR:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64} loc(#loc21)
// CIR:     %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["b", init] {alignment = 4 : i64} loc(#loc22)
// CIR:     %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64} loc(#loc2)
// CIR:     %3 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64} loc(#loc23)
// CIR:     cir.store %arg0, %0 : !s32i, cir.ptr <!s32i> loc(#loc9)
// CIR:     cir.store %arg1, %1 : !s32i, cir.ptr <!s32i> loc(#loc9)
// CIR:     %4 = cir.load %0 : cir.ptr <!s32i>, !s32i loc(#loc10)
// CIR:     %5 = cir.load %1 : cir.ptr <!s32i>, !s32i loc(#loc8)
// CIR:     %6 = cir.binop(add, %4, %5) : !s32i loc(#loc24)
// CIR:     cir.store %6, %3 : !s32i, cir.ptr <!s32i> loc(#loc23)
// CIR:     cir.scope {
// CIR:       %9 = cir.load %3 : cir.ptr <!s32i>, !s32i loc(#loc13)
// CIR:       %10 = cir.const(#cir.int<0> : !s32i) : !s32i loc(#loc14)
// CIR:       %11 = cir.cmp(gt, %9, %10) : !s32i, !cir.bool loc(#loc26)
// CIR:       cir.if %11 {
// CIR:         %12 = cir.const(#cir.int<0> : !s32i) : !s32i loc(#loc16)
// CIR:         cir.store %12, %3 : !s32i, cir.ptr <!s32i> loc(#loc28)
// CIR:       } else {
// CIR:         %12 = cir.const(#cir.int<1> : !s32i) : !s32i loc(#loc12)
// CIR:         cir.store %12, %3 : !s32i, cir.ptr <!s32i> loc(#loc29)
// CIR:       } loc(#loc27)
// CIR:     } loc(#loc25)
// CIR:     %7 = cir.load %3 : cir.ptr <!s32i>, !s32i loc(#loc18)
// CIR:     cir.store %7, %2 : !s32i, cir.ptr <!s32i> loc(#loc30)
// CIR:     %8 = cir.load %2 : cir.ptr <!s32i>, !s32i loc(#loc30)
// CIR:     cir.return %8 : !s32i loc(#loc30)
// CIR:   } loc(#loc20)
// CIR: } loc(#loc)
// CIR: #loc = loc("{{.*}}sourcelocation.cpp":0:0)
// CIR: #loc1 = loc("{{.*}}sourcelocation.cpp":6:1)
// CIR: #loc2 = loc("{{.*}}sourcelocation.cpp":13:1)
// CIR: #loc7 = loc("{{.*}}sourcelocation.cpp":7:3)
// CIR: #loc8 = loc("{{.*}}sourcelocation.cpp":7:15)
// CIR: #loc9 = loc("{{.*}}sourcelocation.cpp":6:22)
// CIR: #loc10 = loc("{{.*}}sourcelocation.cpp":7:11)
// CIR: #loc11 = loc("{{.*}}sourcelocation.cpp":8:3)
// CIR: #loc12 = loc("{{.*}}sourcelocation.cpp":11:9)
// CIR: #loc13 = loc("{{.*}}sourcelocation.cpp":8:7)
// CIR: #loc14 = loc("{{.*}}sourcelocation.cpp":8:11)
// CIR: #loc15 = loc("{{.*}}sourcelocation.cpp":9:5)
// CIR: #loc16 = loc("{{.*}}sourcelocation.cpp":9:9)
// CIR: #loc17 = loc("{{.*}}sourcelocation.cpp":11:5)
// CIR: #loc18 = loc("{{.*}}sourcelocation.cpp":12:10)
// CIR: #loc19 = loc("{{.*}}sourcelocation.cpp":12:3)
// CIR: #loc20 = loc(fused[#loc1, #loc2])
// CIR: #loc23 = loc(fused[#loc7, #loc8])
// CIR: #loc24 = loc(fused[#loc10, #loc8])
// CIR: #loc25 = loc(fused[#loc11, #loc12])
// CIR: #loc26 = loc(fused[#loc13, #loc14])
// CIR: #loc27 = loc(fused[#loc15, #loc16, #loc17, #loc12])
// CIR: #loc28 = loc(fused[#loc15, #loc16])
// CIR: #loc29 = loc(fused[#loc17, #loc12])
// CIR: #loc30 = loc(fused[#loc19, #loc18])


// LLVM: ModuleID = '{{.*}}sourcelocation.cpp'
// LLVM: source_filename = "{{.*}}sourcelocation.cpp"
// LLVM: define i32 @_Z2s0ii(i32 %0, i32 %1) #[[#]] !dbg ![[#SP:]]
// LLVM:  %3 = alloca i32, i64 1, align 4, !dbg ![[#LOC1:]]


// LLVM: !llvm.module.flags = !{!0}
// LLVM: !llvm.dbg.cu = !{!1}
// LLVM: !0 = !{i32 2, !"Debug Info Version", i32 3}
// LLVM: !1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
// LLVM: !2 = !DIFile(filename: "sourcelocation.cpp", directory: "{{.*}}CodeGen")
// LLVM: ![[#SP]] = distinct !DISubprogram(name: "_Z2s0ii", linkageName: "_Z2s0ii", scope: !2, file: !2, line: 6, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
// LLVM: ![[#LOC1]] = !DILocation(line: 6, scope: ![[#SP]])
