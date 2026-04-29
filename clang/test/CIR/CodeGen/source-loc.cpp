// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

namespace std {
  struct source_location {
    struct __impl {
    const char * _M_file_name;
    const char * _M_function_name;
    unsigned _M_line;
    unsigned _M_column;
    };

    const __impl * _impl = nullptr;

    static constexpr source_location
    current(const __impl *p = __builtin_source_location()) {
      source_location loc;
      loc._impl = p;
      return loc;
    }
  };
}

// CIR: cir.global "private" constant cir_private dso_local @".str" = #cir.const_array<
// LLVM: @.str = private constant 
// OGCG: @.str = private unnamed_addr constant 
//
// CIR: cir.global "private" constant cir_private dso_local @".str.1" = #cir.const_array<"void use1()" : !cir.array<!s8i x 11>, trailing_zeros> : !cir.array<!s8i x 12>
// LLVM: @.str.1 = private constant [{{.*}} x i8] c"void use1
// OGCG: @.str.1 = private unnamed_addr constant [{{.*}} x i8] c"void use1
//
// CIR: cir.global "private" constant cir_private @".constant" = #cir.const_record<{#cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.global_view<@".str.1"> : !cir.ptr<!s8i>, #cir.int<{{.*}}> : !u32i, #cir.int<{{.*}}> : !u32i}> : !rec_std3A3Asource_location3A3A__impl
// LLVM: @.constant = private constant %"struct.std::source_location::__impl" { ptr @.str, ptr @.str.1, i32 {{.*}}, i32 {{.*}} }
// OGCG: @.constant = private unnamed_addr constant %"struct.std::source_location::__impl" { ptr @.str, ptr @.str.1, i32 {{.*}}, i32 {{.*}} }
//
// CIR: cir.global "private" constant cir_private dso_local @".str.2" = #cir.const_array<"void use2()" : !cir.array<!s8i x 11>, trailing_zeros> : !cir.array<!s8i x 12> {alignment = 1 : i64} loc(#loc1)
// LLVM: @.str.2 = private constant [{{.*}} x i8] c"void use2
// OGCG: @.str.2 = private unnamed_addr constant [{{.*}} x i8] c"void use2
//
// Note: the naming difference between LLVM and OGCG here is because of the
// uniquification differences when we encounter duplicate names. LLVM has a
// global counter that manages these, CIR just increments the single names, each
// with its own counter.  As a result, .str and .constant names don't match.
// CIR: cir.global "private" constant cir_private @".constant.1" = #cir.const_record<{#cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.global_view<@".str.2"> : !cir.ptr<!s8i>, #cir.int<{{.*}}> : !u32i, #cir.int<{{.*}}> : !u32i}> : !rec_std3A3Asource_location3A3A__impl
// LLVM: @.constant.1 = private constant %"struct.std::source_location::__impl" { ptr @.str, ptr @.str.2, i32 {{.*}}, i32 {{.*}} }
// OGCG: @.constant.3 = private unnamed_addr constant %"struct.std::source_location::__impl" { ptr @.str, ptr @.str.2, i32 {{.*}}, i32 {{.*}} }


void has_sl(std::source_location loc = std::source_location::current());
void use1() { has_sl(); }
// CIR-LABEL: cir.func{{.*}} @_Z4use1v
// CIR: cir.const #cir.global_view<@".constant"> : !cir.ptr<!rec_std3A3Asource_location3A3A__impl>
// LLVM: define {{.*}}@_Z4use1v()
// LLVM: call {{.*}}@_ZNSt15source_location7currentEPKNS_6__implE(ptr noundef @.constant)
// OGCG: call {{.*}}@_ZNSt15source_location7currentEPKNS_6__implE(ptr noundef @.constant)
void use2() { has_sl(); }
// CIR-LABEL: cir.func{{.*}} @_Z4use2v
// CIR: cir.const #cir.global_view<@".constant.1"> : !cir.ptr<!rec_std3A3Asource_location3A3A__impl>
// LLVM: define {{.*}}@_Z4use2v()
// LLVM: call {{.*}}@_ZNSt15source_location7currentEPKNS_6__implE(ptr noundef @.constant.1)
// OGCG: call {{.*}}@_ZNSt15source_location7currentEPKNS_6__implE(ptr noundef @.constant.3)

void line_column() {
  unsigned int a = __builtin_LINE();
  unsigned int b = __builtin_COLUMN();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init]
// CIR: %[[CONST_9:.*]] = cir.const #cir.int<68> : !u32i
// CIR: cir.store {{.*}} %[[CONST_9]], %[[A_ADDR]] : !u32i, !cir.ptr<!u32i>
// CIR: %[[CONST_20:.*]] = cir.const #cir.int<20> : !u32i
// CIR: cir.store {{.*}} %[[CONST_20]], %[[B_ADDR]] : !u32i, !cir.ptr<!u32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 68, ptr %[[A_ADDR]], align 4
// LLVM: store i32 20, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 68, ptr %[[A_ADDR]], align 4
// OGCG: store i32 20, ptr %[[B_ADDR]], align 4

void function_file() {
  const char *a = __builtin_FUNCTION();
  const char *b = __builtin_FILE();
  const char *c = __builtin_FILE_NAME();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["c", init]
// CIR: %[[FUNC__GV:.*]] = cir.const #cir.global_view<@".str.3"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FUNC__GV]], %[[A_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: %[[FILE_PATH_GV:.*]] = cir.const #cir.global_view<@".str"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FILE_PATH_GV]], %[[B_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: %[[FILE_GV:.*]] = cir.const #cir.global_view<@".str.4"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FILE_GV]], %[[C_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr @.str.3, ptr %[[A_ADDR]], align 8
// LLVM: store ptr @.str, ptr %[[B_ADDR]], align 8
// LLVM: store ptr @.str.4, ptr %[[C_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[B_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[C_ADDR:.*]] = alloca ptr, align 8
// OGCG: store ptr @.str.4, ptr %[[A_ADDR]], align 8
// OGCG: store ptr @.str, ptr %[[B_ADDR]], align 8
// OGCG: store ptr @.str.5, ptr %[[C_ADDR]], align 8
