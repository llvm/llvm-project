// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct View {
  const char *ptr;
  int len;
  constexpr View(const char *p, int n) : ptr(p), len(n) {}
};

constexpr const char *global_str = "hello";

void test() {
  constexpr View v(global_str + 2, 3);
  (void)v;
}

// CIR: cir.global "private" constant cir_private @__const._Z4testv.v = #cir.const_record<{#cir.global_view<@"[[STR_NAME:.*]]", [2 : i32]> : !cir.ptr<!s8i>, #cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_View
// CIR: cir.global "private" constant cir_private dso_local @"[[STR_NAME]]" = #cir.const_array<"hello" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6>
// LLVMCIR: @__const._Z4testv.v = private constant %struct.View <{ ptr getelementptr inbounds nuw (i8, ptr @[[STR_NAME:.*]], i64 2), i32 3, [4 x i8] zeroinitializer }>
// LLVMCIR: @[[STR_NAME]] = private {{.*}}constant [6 x i8] c"hello\00"

// OGCG: @[[STR_NAME:.*]] = private {{.*}}constant [6 x i8] c"hello\00"
// OGCG: @__const._Z4testv.v = private unnamed_addr constant { ptr, i32 } { ptr getelementptr (i8, ptr @.str, i64 2), i32 3 }

// CIR-LABEL: @_Z4testv
// CIR: cir.get_global @__const._Z4testv.v : !cir.ptr<!rec_View>

// LLVM-LABEL: @_Z4testv
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}@__const._Z4testv.v, i64 16, i1 false)
