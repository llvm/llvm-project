// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1 -no-pointer-tbaa
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -no-pointer-tbaa
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1 -pointer-tbaa
// RUN: FileCheck --check-prefix=CIR-POINTER-TBAA --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -pointer-tbaa
// RUN: FileCheck --check-prefix=LLVM-POINTER-TBAA --input-file=%t.ll %s

// CIR: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<id = "int", type = !s32i>
// CIR: #tbaa[[PTR_TO_A:.*]] = #cir.tbaa_scalar<id = "any pointer", type = !cir.ptr<!rec_A>>
// CIR: #tbaa[[STRUCT_A:.*]] = #cir.tbaa_struct<id = "_ZTS1A", members = {<#tbaa[[INT]], 0>, <#tbaa[[INT]], 4>}>
// CIR: #tbaa[[TAG_STRUCT_A_a:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_A]], access = #tbaa[[INT]], offset = 0>

// CIR-POINTER-TBAA: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR-POINTER-TBAA: #tbaa[[INT:.*]] = #cir.tbaa_scalar<id = "int", type = !s32i>
// CIR-POINTER-TBAA-DAG: #tbaa[[p1_INT:.*]] = #cir.tbaa_scalar<id = "p1 int", type = !cir.ptr<!s32i>
// CIR-POINTER-TBAA-DAG: #tbaa[[p2_INT:.*]] = #cir.tbaa_scalar<id = "p2 int", type = !cir.ptr<!cir.ptr<!s32i>>
// CIR-POINTER-TBAA-DAG: #tbaa[[p3_INT:.*]] = #cir.tbaa_scalar<id = "p3 int", type = !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR-POINTER-TBAA-DAG: #tbaa[[STRUCT_A:.*]] = #cir.tbaa_struct<id = "_ZTS1A", members = {<#tbaa[[INT]], 0>, <#tbaa[[INT]], 4>}>
// CIR-POINTER-TBAA-DAG: #tbaa[[p1_STRUCT_A:.*]] = #cir.tbaa_scalar<id = "p1 _ZTS1A", type = !cir.ptr<!rec_A>
// CIR-POINTER-TBAA-DAG: #tbaa[[p2_STRUCT_A:.*]] = #cir.tbaa_scalar<id = "p2 _ZTS1A", type = !cir.ptr<!cir.ptr<!rec_A>>
// CIR-POINTER-TBAA-DAG: #tbaa[[p3_STRUCT_A:.*]] = #cir.tbaa_scalar<id = "p3 _ZTS1A", type = !cir.ptr<!cir.ptr<!cir.ptr<!rec_A>>>

int test_scalar_pointer(int*** p3) {
    int* p1;
    int** p2;
    p2 = *p3;
    p1 = *p2;
    int t = *p1;

    // CIR-POINTER-TBAA-LABEL: _Z19test_scalar_pointerPPPi
    // CIR-POINTER-TBAA: %{{.*}} = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> tbaa(#tbaa[[p3_INT]])
    // CIR-POINTER-TBAA: %{{.*}} = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>> tbaa(#tbaa[[p2_INT]])
    // CIR-POINTER-TBAA: %{{.*}} = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> tbaa(#tbaa[[p1_INT]])

    // LLVM-LABEL: _Z19test_scalar_pointerPPPi
    // LLVM: %[[p2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_ANY_PTR:.*]]
    // LLVM: %[[p1:.*]] = load ptr, ptr %[[p2]], align 8, !tbaa ![[TBAA_ANY_PTR]]
    // LLVM: %[[t:.*]] = load i32, ptr %[[p1]], align 4, !tbaa ![[TBAA_INT:.*]]

    // LLVM-POINTER-TBAA-LABEL: _Z19test_scalar_pointerPPPi
    // LLVM-POINTER-TBAA: %[[p2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_p2_INT:.*]]
    // LLVM-POINTER-TBAA: %[[p1:.*]] = load ptr, ptr %[[p2]], align 8, !tbaa ![[TBAA_p1_INT:.*]]
    // LLVM-POINTER-TBAA: %[[t:.*]] = load i32, ptr %[[p1]], align 4, !tbaa ![[TBAA_INT:.*]]
    return t;
}

struct A {
    int a;
    int b;
};

int test_struct_pointer(A*** p3, int A::***m3) {
    A* p1;
    A** p2;
    p2 = *p3;
    p1 = *p2;

    // CIR-POINTER-TBAA-LABEL: _Z19test_struct_pointerPPP1APPMS_i
    // CIR-POINTER-TBAA: %{{.*}} = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_A>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_A>>> tbaa(#tbaa[[p3_STRUCT_A]])
    // CIR-POINTER-TBAA: %{{.*}} = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.ptr<!rec_A>>>, !cir.ptr<!cir.ptr<!rec_A>> tbaa(#tbaa[[p2_STRUCT_A]])
    // CIR-POINTER-TBAA: %{{.*}} = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A> tbaa(#tbaa[[p1_STRUCT_A]])

    // LLVM-LABEL: _Z19test_struct_pointerPPP1APPMS_i
    // LLVM: %[[p2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_ANY_PTR]]
    // LLVM: %[[p1:.*]] = load ptr, ptr %[[p2]], align 8, !tbaa ![[TBAA_ANY_PTR]]
    // LLVM: %[[t:.*]] = load i32, ptr %[[p1]], align 4, !tbaa ![[TBAA_STRUCT_A_a:.*]]

    // LLVM-POINTER-TBAA-LABEL: _Z19test_struct_pointerPPP1APPMS_i
    // LLVM-POINTER-TBAA: %[[p2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_p2_STRUCT_A:.*]]
    // LLVM-POINTER-TBAA: %[[p1:.*]] = load ptr, ptr %[[p2]], align 8, !tbaa ![[TBAA_p1_STRUCT_A:.*]]
    // LLVM-POINTER-TBAA: %[[t:.*]] = load i32, ptr %[[p1]], align 4, !tbaa ![[TBAA_STRUCT_A_a:.*]]
    return p1->a;
}

void test_member_pointer(A& a, int A::***m3, int val) {

    // CIR-LABEL: _Z19test_member_pointerR1APPMS_ii
    // CIR: %{{.*}} = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.data_member<!s32i in !rec_A>>, !cir.data_member<!s32i in !rec_A> tbaa(#tbaa[[CHAR]])

    // CIR-POINTER-TBAA-LABEL: _Z19test_member_pointerR1APPMS_ii
    // CIR-POINTER-TBAA: %{{.*}} = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.data_member<!s32i in !rec_A>>, !cir.data_member<!s32i in !rec_A> tbaa(#tbaa[[CHAR]])

    // LLVM-LABEL: _Z19test_member_pointerR1APPMS_ii
    // LLVM: %[[m2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_ANY_PTR:.*]]
    // LLVM: %[[m1:.*]] = load i64, ptr %[[m2]], align 8, !tbaa ![[TBAA_member_ptr:.*]]
    // LLVM: %[[A_a:.*]] = getelementptr i8, ptr %{{.*}}, i64 %[[m1]]
    // LLVM: store i32 %{{.*}}, ptr %[[A_a]], align 4, !tbaa ![[TBAA_INT]]

    // LLVM-POINTER-TBAA-LABEL: _Z19test_member_pointerR1APPMS_ii
    // LLVM-POINTER-TBAA: %[[m2:.*]] = load ptr, ptr %{{.*}}, align 8, !tbaa ![[TBAA_ANY_PTR:.*]]
    // LLVM-POINTER-TBAA: %[[m1:.*]] = load i64, ptr %[[m2]], align 8, !tbaa ![[TBAA_member_ptr:.*]]
    // LLVM-POINTER-TBAA: %[[A_a:.*]] = getelementptr i8, ptr %{{.*}}, i64 %[[m1]]
    // LLVM-POINTER-TBAA: store i32 %{{.*}}, ptr %[[A_a]], align 4, !tbaa ![[TBAA_INT]]
    a.***m3 = val; 
}

// LLVM: ![[TBAA_ANY_PTR]] = !{![[TBAA_ANY_PTR_PARENT:.*]], ![[TBAA_ANY_PTR_PARENT]], i64 0}
// LLVM: ![[TBAA_ANY_PTR_PARENT]] = !{!"any pointer", ![[CHAR:.*]], i64 0}
// LLVM: ![[CHAR]] = !{!"omnipotent char", ![[ROOT:.*]], i64 0}
// LLVM: ![[ROOT]] = !{!"Simple C++ TBAA"}
// LLVM: ![[TBAA_INT]] = !{![[TBAA_INT_PARENT:.*]], ![[TBAA_INT_PARENT]], i64 0}
// LLVM: ![[TBAA_INT_PARENT]] = !{!"int", ![[CHAR]], i64 0}
// LLVM: ![[TBAA_STRUCT_A_a]] = !{![[TBAA_STRUCT_A:.*]], ![[TBAA_INT_PARENT]], i64 0}
// LLVM: ![[TBAA_STRUCT_A]] = !{!"_ZTS1A", ![[TBAA_INT_PARENT]], i64 0, ![[TBAA_INT_PARENT]], i64 4}
// LLVM: ![[TBAA_member_ptr]] = !{![[CHAR]], ![[CHAR]], i64 0}

// LLVM-POINTER-TBAA: ![[TBAA_p2_INT]] = !{![[TBAA_p2_INT_PARENT:.*]], ![[TBAA_p2_INT_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p2_INT_PARENT]] = !{!"p2 int", ![[TBAA_ANY_PTR_PARENT:.*]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_ANY_PTR_PARENT]] = !{!"any pointer", ![[CHAR:.*]], i64 0}
// LLVM-POINTER-TBAA: ![[CHAR]] = !{!"omnipotent char", ![[ROOT:.*]], i64 0}
// LLVM-POINTER-TBAA: ![[ROOT]] = !{!"Simple C++ TBAA"}
// LLVM-POINTER-TBAA: ![[TBAA_p1_INT]] = !{![[TBAA_p1_INT_PARENT:.*]], ![[TBAA_p1_INT_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p1_INT_PARENT]] = !{!"p1 int", ![[TBAA_ANY_PTR_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_INT]] = !{![[TBAA_INT_PARENT:.*]], ![[TBAA_INT_PARENT]], i64 
// LLVM-POINTER-TBAA: ![[TBAA_INT_PARENT]] = !{!"int", ![[CHAR]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p2_STRUCT_A]] = !{![[TBAA_p2_STRUCT_A_PARENT:.*]], ![[TBAA_p2_STRUCT_A_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p2_STRUCT_A_PARENT]] = !{!"p2 _ZTS1A", ![[TBAA_ANY_PTR_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p1_STRUCT_A]] = !{![[TBAA_p1_STRUCT_A_PARENT:.*]], ![[TBAA_p1_STRUCT_A_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_p1_STRUCT_A_PARENT]] = !{!"p1 _ZTS1A", ![[TBAA_ANY_PTR_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_STRUCT_A_a]] = !{![[TBAA_STRUCT_A:.*]], ![[TBAA_INT_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_STRUCT_A]] = !{!"_ZTS1A", ![[TBAA_INT_PARENT]], i64 0, ![[TBAA_INT_PARENT]], i64 4}
// LLVM-POINTER-TBAA: ![[TBAA_ANY_PTR]] = !{![[TBAA_ANY_PTR_PARENT]], ![[TBAA_ANY_PTR_PARENT]], i64 0}
// LLVM-POINTER-TBAA: ![[TBAA_member_ptr]] = !{![[CHAR]], ![[CHAR]], i64 0}
