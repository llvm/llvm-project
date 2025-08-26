// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR: cir.global "private" constant cir_private dso_local @".str" = #cir.const_array<"%s\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3> 
// CIR: cir.global "private" constant cir_private dso_local @".str.1" = #cir.const_array<"%s %d\0A\00" : !cir.array<!s8i x 7>> : !cir.array<!s8i x 7>
// LLVM: @.str = private constant [3 x i8] c"%s\00"
// LLVM: @.str.1 = private constant [7 x i8] c"%s %d\0A\00"
// OGCG: @.str = private unnamed_addr constant [3 x i8] c"%s\00"
// OGCG: @.str.1 = private unnamed_addr constant [7 x i8] c"%s %d\0A\00"

void func(char const * const str, int i) {
  __builtin_printf(nullptr);
  __builtin_printf("%s", str);
  __builtin_printf("%s %d\n", str, i);
}

// CIR: cir.func{{.*}} @printf(!cir.ptr<!s8i>, ...) -> !s32i

// CIR: cir.func{{.*}} @_Z4funcPKci(%[[arg0:.+]]: !cir.ptr<!s8i>{{.*}}, %[[arg1:.+]]: !s32i{{.*}}) {
// CIR:   %[[str_ptr:.+]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["str", init, const]
// CIR:   %[[i_ptr:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   cir.store %[[arg0]], %[[str_ptr]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR:   cir.store %[[arg1]], %[[i_ptr]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[null_ptr:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i>
// CIR:   %[[printf_result1:.+]] = cir.call @printf(%[[null_ptr]]) nothrow : (!cir.ptr<!s8i>) -> !s32i
// CIR:   %[[str_fmt_global:.+]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>>
// CIR:   %[[str_fmt_ptr:.+]] = cir.cast(array_to_ptrdecay, %[[str_fmt_global]] : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CIR:   %[[str_val:.+]] = cir.load{{.*}} %[[str_ptr]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR:   %[[printf_result2:.+]] = cir.call @printf(%[[str_fmt_ptr]], %[[str_val]]) nothrow : (!cir.ptr<!s8i>, !cir.ptr<!s8i>) -> !s32i
// CIR:   %[[full_fmt_global:.+]] = cir.get_global @".str.1" : !cir.ptr<!cir.array<!s8i x 7>>
// CIR:   %[[full_fmt_ptr:.+]] = cir.cast(array_to_ptrdecay, %[[full_fmt_global]] : !cir.ptr<!cir.array<!s8i x 7>>), !cir.ptr<!s8i>
// CIR:   %[[str_val2:.+]] = cir.load{{.*}} %[[str_ptr]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR:   %[[i_val:.+]] = cir.load{{.*}} %[[i_ptr]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[printf_result3:.+]] = cir.call @printf(%[[full_fmt_ptr]], %[[str_val2]], %[[i_val]]) nothrow : (!cir.ptr<!s8i>, !cir.ptr<!s8i>, !s32i) -> !s32i
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z4funcPKci(ptr %[[arg0:.+]], i32 %[[arg1:.+]])
// LLVM:   %[[str_ptr:.+]] = alloca ptr
// LLVM:   %[[i_ptr:.+]] = alloca i32
// LLVM:   store ptr %[[arg0]], ptr %[[str_ptr]]{{.*}}
// LLVM:   store i32 %[[arg1]], ptr %[[i_ptr]]{{.*}}
// LLVM:   %[[printf_result1:.+]] = call i32 (ptr, ...) @printf(ptr null)
// LLVM:   %[[str_val:.+]] = load ptr, ptr %[[str_ptr]]{{.*}}
// LLVM:   %[[printf_result2:.+]] = call i32 (ptr, ...) @printf(ptr @.str, ptr %[[str_val]])
// LLVM:   %[[str_val2:.+]] = load ptr, ptr %[[str_ptr]]{{.*}}
// LLVM:   %[[i_val:.+]] = load i32, ptr %[[i_ptr]]{{.*}}
// LLVM:   %[[printf_result3:.+]] = call i32 (ptr, ...) @printf(ptr @.str.1, ptr %[[str_val2]], i32 %[[i_val]])
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z4funcPKci(ptr noundef %[[arg0:.+]], i32 noundef %[[arg1:.+]])
// OGCG:   %[[str_ptr:.+]] = alloca ptr
// OGCG:   %[[i_ptr:.+]] = alloca i32
// OGCG:   store ptr %[[arg0]], ptr %[[str_ptr]]{{.*}}
// OGCG:   store i32 %[[arg1]], ptr %[[i_ptr]]{{.*}}
// OGCG:   %[[printf_result1:.+]] = call i32 (ptr, ...) @printf(ptr noundef null)
// OGCG:   %[[str_val:.+]] = load ptr, ptr %[[str_ptr]]{{.*}}
// OGCG:   %[[printf_result2:.+]] = call i32 (ptr, ...) @printf(ptr noundef @.str, ptr noundef %[[str_val]])
// OGCG:   %[[str_val2:.+]] = load ptr, ptr %[[str_ptr]]{{.*}}
// OGCG:   %[[i_val:.+]] = load i32, ptr %[[i_ptr]]{{.*}}
// OGCG:   %[[printf_result3:.+]] = call i32 (ptr, ...) @printf(ptr noundef @.str.1, ptr noundef %[[str_val2]], i32 noundef %[[i_val]])
// OGCG:   ret void
