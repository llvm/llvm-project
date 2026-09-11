// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

void take_wide(_BitInt(200) x) {}

// CIR: cir.func{{.*}} @take_wide(
// CIR-SAME:   %arg0: !cir.ptr<!cir.int<s, 200, bitint>> {llvm.align = 8 : i64
// CIR-SAME:   , llvm.byval = !cir.int<s, 200, bitint>, llvm.noundef} loc{{.*}})

// The byval copy is sized from the padded storage type (i256), matching
// classic.  One divergence survives: the position of the local's alloca,
// which CIR places at the point of the declaration while classic hoists
// every alloca to the function's entry block ahead of the parameter
// materialization.
// LLVM-LABEL: define dso_local void @take_wide(
// LLVM-SAME:     ptr noundef byval(i256) align 8 %[[ARG:[0-9]+]])

// LLVMCIR-NEXT:  %[[WIDE:.+]] = load i256, ptr %[[ARG]], align 8
// LLVMCIR-NEXT:  %[[NARROW:.+]] = trunc i256 %[[WIDE]] to i200
// LLVMCIR-NEXT:  %[[SLOT:.+]] = alloca i256, align 8
// LLVMCIR-NEXT:  %[[EXT:.+]] = sext i200 %[[NARROW]] to i256
// LLVMCIR-NEXT:  store i256 %[[EXT]], ptr %[[SLOT]], align 8
// LLVMCIR-NEXT:  ret void

// OGCG:          %[[SLOT:.+]] = alloca i256, align 8
// OGCG-NEXT:     %[[WIDE:.+]] = load i256, ptr %[[ARG]], align 8
// OGCG-NEXT:     %[[NARROW:.+]] = trunc i256 %[[WIDE]] to i200
// OGCG-NEXT:     %[[EXT:.+]] = sext i200 %[[NARROW]] to i256
// OGCG-NEXT:     store i256 %[[EXT]], ptr %[[SLOT]], align 8
// OGCG-NEXT:     ret void
