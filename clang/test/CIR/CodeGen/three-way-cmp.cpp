// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefix=BEFORE,BOTH
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=AFTER

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER,BOTH

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll 2>&1
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t-og.ll 2>&1
// RUN: FileCheck --input-file=%t-og.ll %s -check-prefix=OGCG

#include "./Inputs/std-compare.h"

// BEFORE: #cmpinfo_partial_ltn1eq0gt1unn127 = #cir.cmp3way_info<partial, lt = -1, eq = 0, gt = 1, unordered = -127>
// BEFORE: #cmpinfo_strong_ltn1eq0gt1 = #cir.cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
// BEFORE: !rec_std3A3A__13A3Apartial_ordering = !cir.record<class "std::__1::partial_ordering" {!s8i}>
// BEFORE: !rec_std3A3A__13A3Astrong_ordering = !cir.record<class "std::__1::strong_ordering" {!s8i}>

auto three_way_strong(int x, int y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z16three_way_strongii
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_strong_ltn1eq0gt1 %{{.+}}, %{{.+}} : !s32i -> !s8i
// BEFORE: }

//      AFTER:   cir.func {{.*}} @_Z16three_way_strongii{{.*}}
//      AFTER:   %[[LHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i{{.*}}
// AFTER-NEXT:   %[[RHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i{{.*}}
// AFTER-NEXT:   %[[LT:.*]] = cir.const #cir.int<-1> : !s8i{{.*}}
// AFTER-NEXT:   %[[EQ:.*]] = cir.const #cir.int<0> : !s8i{{.*}}
// AFTER-NEXT:   %[[GT:.*]] = cir.const #cir.int<1> : !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_LT:.*]] = cir.cmp lt %[[LHS]], %[[RHS]] : !s32i{{.*}}
// AFTER-NEXT:   %[[SELECT_1:.*]] = cir.select if %[[CMP_LT]] then %[[LT]] else %[[GT]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_EQ:.*]] = cir.cmp eq %[[LHS]], %[[RHS]] : !s32i{{.*}}
// AFTER-NEXT:   %[[SELECT_2:.*]] = cir.select if %[[CMP_EQ]] then %[[EQ]] else %[[SELECT_1]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER:   %{{.+}} = cir.load %{{.+}} : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, !rec_std3A3A__13A3Astrong_ordering{{.*}}
// AFTER-NEXT:   cir.return %{{.+}} : !rec_std3A3A__13A3Astrong_ordering{{.*}}

// LLVM:  %[[LHS:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-NEXT:  %[[RHS:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-NEXT:  %[[CMP_LT:.*]] = icmp slt i32 %[[LHS]], %[[RHS]]
// LLVM-NEXT:  %[[SEL_LT_GT:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 1
// LLVM-NEXT:  %[[CMP_EQ:.*]] = icmp eq i32 %[[LHS]], %[[RHS]]
// LLVM-NEXT:  %[[RES:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 %[[SEL_LT_GT]]

// OGCG:  %[[LHS:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG-NEXT:  %[[RHS:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG-NEXT:  %[[CMP_LT:.*]] = icmp slt i32 %[[LHS]], %[[RHS]]
// OGCG-NEXT:  %[[SEL_EQ_LT:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 1
// OGCG-NEXT:  %[[CMP_EQ:.*]] = icmp eq i32 %[[LHS]], %[[RHS]]
// OGCG-NEXT:  %[[RES:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 %[[SEL_EQ_LT]]

auto three_way_partial(float x, float y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z17three_way_partialff
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_partial_ltn1eq0gt1unn127 %{{.+}}, %{{.+}} : !cir.float -> !s8i
// BEFORE: }

//      AFTER:   cir.func {{.*}} @_Z17three_way_partialff{{.*}}
//      AFTER:   %[[LHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!cir.float>, !cir.float{{.*}}
// AFTER-NEXT:   %[[RHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!cir.float>, !cir.float{{.*}}
// AFTER-NEXT:   %[[LT:.*]] = cir.const #cir.int<-1> : !s8i{{.*}}
// AFTER-NEXT:   %[[EQ:.*]] = cir.const #cir.int<0> : !s8i{{.*}}
// AFTER-NEXT:   %[[GT:.*]] = cir.const #cir.int<1> : !s8i{{.*}}
// AFTER-NEXT:   %[[UNORDERED:.*]] = cir.const #cir.int<-127> : !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_EQ:.*]] = cir.cmp eq %[[LHS]], %[[RHS]] : !cir.float{{.*}}
// AFTER-NEXT:   %[[SELECT_1:.*]] = cir.select if %[[CMP_EQ]] then %[[EQ]] else %[[UNORDERED]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_GT:.*]] = cir.cmp gt %[[LHS]], %[[RHS]] : !cir.float{{.*}}
// AFTER-NEXT:   %[[SELECT_2:.*]] = cir.select if %[[CMP_GT]] then %[[GT]] else %[[SELECT_1]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_LT:.*]] = cir.cmp lt %[[LHS]], %[[RHS]] : !cir.float{{.*}}
// AFTER-NEXT:   %[[SELECT_3:.*]] = cir.select if %[[CMP_LT]] then %[[LT]] else %[[SELECT_2]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER:   %{{.+}} = cir.load %{{.+}} : !cir.ptr<!rec_std3A3A__13A3Apartial_ordering>, !rec_std3A3A__13A3Apartial_ordering{{.*}}
// AFTER-NEXT:   cir.return %{{.+}} : !rec_std3A3A__13A3Apartial_ordering{{.*}}

// LLVM:  %[[LHS:.*]] = load float, ptr %{{.*}}, align 4
// LLVM:  %[[RHS:.*]] = load float, ptr %{{.*}}, align 4
// LLVM:  %[[CMP_EQ:.*]] = fcmp oeq float %[[LHS]], %[[RHS]]
// LLVM:  %[[SEL_EQ_UN:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 -127
// LLVM:  %[[CMP_GT:.*]] = fcmp ogt float %[[LHS]], %[[RHS]]
// LLVM:  %[[SEL_GT_EQUN:.*]] = select i1 %[[CMP_GT]], i8 1, i8 %[[SEL_EQ_UN]]
// LLVM:  %[[CMP_LT:.*]] = fcmp olt float %[[LHS]], %[[RHS]]
// LLVM:  %[[RES:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_GT_EQUN]]

// OGCG:  %[[LHS:.*]] = load float, ptr %{{.*}}, align 4
// OGCG:  %[[RHS:.*]] = load float, ptr %{{.*}}, align 4
// OGCG:  %[[CMP_EQ:.*]] = fcmp oeq float %[[LHS]], %[[RHS]]
// OGCG:  %[[SEL_EQ_UN:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 -127
// OGCG:  %[[CMP_GT:.*]] = fcmp ogt float %[[LHS]], %[[RHS]]
// OGCG:  %[[SEL_GT_EQUN:.*]] = select i1 %[[CMP_GT]], i8 1, i8 %[[SEL_EQ_UN]]
// OGCG:  %[[CMP_LT:.*]] = fcmp olt float %[[LHS]], %[[RHS]]
// OGCG:  %[[RES:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_GT_EQUN]]

struct Member {
  bool operator==(const Member&) const;
  bool operator<(const Member&) const;
};

struct HasMember {
  Member m;
  std::strong_ordering operator<=>(const HasMember&) const = default;
};

void use_pseudo_ordering(HasMember m1, HasMember m2) {
  // BOTH: cir.func {{.*}}@_ZNK9HasMemberssERKS_(%{{.*}}: !cir.ptr<!rec_HasMember>{{.*}}, %{{.*}}: !cir.ptr<!rec_HasMember>{{.*}}) -> !rec_std3A3A__13A3Astrong_ordering
  // BOTH: %[[LHS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasMember>, !cir.ptr<!cir.ptr<!rec_HasMember>>, ["this", init]
  // BOTH: %[[RHS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasMember>, !cir.ptr<!cir.ptr<!rec_HasMember>>, ["", init, const]
  // BOTH: %[[RET_ALLOCA:.*]] = cir.alloca !rec_std3A3A__13A3Astrong_ordering, !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, ["__retval"]
  // BOTH: %[[LHS_LOAD:.*]] = cir.load deref %[[LHS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasMember>>, !cir.ptr<!rec_HasMember>
  // BOTH: cir.scope {
  // BOTH:   %[[CMP_RES:.*]] = cir.alloca !rec_std3A3A__13A3Astrong_ordering, !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, ["cmp", init]
  // BOTH:   %[[CMP_TEMP:.*]] = cir.alloca !rec_std3A3A__13A3Astrong_ordering, !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, ["agg.tmp0"]
  // BOTH:   %[[LHS_MEMBER:.*]] = cir.cast bitcast %[[LHS_LOAD]] : !cir.ptr<!rec_HasMember> -> !cir.ptr<!rec_Member>
  // BOTH:   %[[RHS_LOAD:.*]] = cir.load %[[RHS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasMember>>, !cir.ptr<!rec_HasMember>
  // BOTH:   %[[RHS_MEMBER:.*]] = cir.cast bitcast %[[RHS_LOAD]] : !cir.ptr<!rec_HasMember> -> !cir.ptr<!rec_Member>
  // BOTH:   %[[EQ_RES:.*]] = cir.call @_ZNK6MembereqERKS_(%[[LHS_MEMBER]], %[[RHS_MEMBER]]) : (!cir.ptr<!rec_Member> {{.*}}, !cir.ptr<!rec_Member> {{.*}})
  // BOTH:   %[[TOP_TERN_RES:.*]] = cir.ternary(%[[EQ_RES]], true {
  // BOTH:     %[[EQ_GLOB:.*]] = cir.get_global @_ZNSt3__115strong_ordering5equalE : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:     cir.yield %[[EQ_GLOB]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:   }, false {
  // BOTH:     %[[LT_RES:.*]] = cir.call @_ZNK6MemberltERKS_(%[[LHS_MEMBER]], %[[RHS_MEMBER]]) : (!cir.ptr<!rec_Member> {{.*}}, !cir.ptr<!rec_Member> {{.*}}) -> (!cir.bool {{.*}})
  // BOTH:     %[[TERN_LT_RES:.*]] = cir.ternary(%[[LT_RES]], true {
  // BOTH:       %[[LT_GLOB:.*]] = cir.get_global @_ZNSt3__115strong_ordering4lessE : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:       cir.yield %[[LT_GLOB]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:     }, false {
  // BOTH:       %[[GT_GLOB:.*]] = cir.get_global @_ZNSt3__115strong_ordering7greaterE : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:       cir.yield %[[GT_GLOB]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:     }) : (!cir.bool) -> !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:     cir.yield %[[TERN_LT_RES]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:   }) : (!cir.bool) -> !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:   cir.copy %[[TOP_TERN_RES]] to %[[CMP_RES]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:   cir.copy %[[CMP_RES]] to %[[CMP_TEMP]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:   %[[UNSPEC_TEMP:.*]] = cir.const #cir.const_record<{#cir.int<0> : !s64i, #cir.int<0> : !s64i}>
  // BOTH:   %[[CMP_TEMP_LOAD:.*]] = cir.load {{.*}}%[[CMP_TEMP]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, !rec_std3A3A__13A3Astrong_ordering
  // BOTH:   %[[SO_NE_RES:.*]] = cir.call @_ZNSt3__1neENS_15strong_orderingEMNS_19_CmpUnspecifiedTypeEFvvE(%14, %[[UNSPEC_TEMP]])
  // BOTH:   cir.if %[[SO_NE_RES]] {
  // BOTH:     cir.copy %[[CMP_RES]] to %[[RET_ALLOCA]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH:     %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, !rec_std3A3A__13A3Astrong_ordering
  // BOTH:     cir.return %[[RET_LOAD]] : !rec_std3A3A__13A3Astrong_ordering
  // BOTH:   }
  // BOTH: }
  // BOTH: %[[EQ_GLOB:.*]] = cir.get_global @_ZNSt3__115strong_ordering5equalE : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH: cir.copy %[[EQ_GLOB]] to %[[RET_ALLOCA]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  // BOTH: %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, !rec_std3A3A__13A3Astrong_ordering
  // BOTH: cir.return %[[RET_LOAD]] : !rec_std3A3A__13A3Astrong_ordering

  // BOTH: cir.func {{.*}} @_Z19use_pseudo_ordering9HasMemberS_(%[[M1:.*]]: !rec_HasMember{{.*}}, %[[M2:.*]]: !rec_HasMember{{.*}})
  // BOTH: %[[M1_ALLOCA:.*]] = cir.alloca !rec_HasMember, !cir.ptr<!rec_HasMember>, ["m1", init]
  // BOTH: %[[M2_ALLOCA:.*]] = cir.alloca !rec_HasMember, !cir.ptr<!rec_HasMember>, ["m2", init] {alignment = 1 : i64}
  // BOTH: %[[G_ALLOCA:.*]] = cir.alloca !rec_std3A3A__13A3Astrong_ordering, !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, ["g", init]
  // BOTH: %[[CALL_RES:.*]] = cir.call @_ZNK9HasMemberssERKS_(%[[M1_ALLOCA]], %[[M2_ALLOCA]]) : (!cir.ptr<!rec_HasMember> {{.*}}, !cir.ptr<!rec_HasMember> {{.*}}) -> !rec_std3A3A__13A3Astrong_ordering
  // BOTH: cir.store {{.*}}%[[CALL_RES]], %[[G_ALLOCA]] : !rec_std3A3A__13A3Astrong_ordering, !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>
  std::strong_ordering g = (m1 <=> m2);
  // LLVM: define {{.*}} @_ZNK9HasMemberssERKS_(ptr {{.*}}, ptr {{.*}})
  // LLVM:   %[[TMP_SO:.*]] = alloca %"class.std::__1::strong_ordering"
  // LLVM:   %[[RET_ALLOCA:.*]] = alloca %"class.std::__1::strong_ordering"
  // LLVM:   %[[LHS_ALLOCA:.*]] = alloca ptr
  // LLVM:   %[[RHS_ALLOCA:.*]] = alloca ptr
  // LLVM:   %[[TMP_SO2:.*]] = alloca %"class.std::__1::strong_ordering"
  // LLVM:   %[[LHS_LOAD:.*]] = load ptr, ptr %[[LHS_ALLOCA]]
  //
  // LLVM:   %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
  // LLVM:   %[[EQ_RES:.*]] = call noundef i1 @_ZNK6MembereqERKS_(ptr {{.*}}%[[LHS_LOAD]], ptr {{.*}}%[[RHS_LOAD]])
  // LLVM:   br i1 %[[EQ_RES]], label %[[EQ_TRUE:.*]], label %[[EQ_FALSE:.*]]
  //
  // LLVM: [[EQ_TRUE]]:
  // LLVM:   br label %20
  //
  // LLVM: [[EQ_FALSE]]:
  // LLVM:   %[[LT_RES:.*]] = call noundef i1 @_ZNK6MemberltERKS_(ptr {{.*}}%[[LHS_LOAD]], ptr {{.*}}%[[RHS_LOAD]])
  // LLVM:   br i1 %[[LT_RES]], label %[[LT_TRUE:.*]], label %[[LT_FALSE:.*]]
  //
  // LLVM: [[LT_TRUE]]:
  // LLVM:   br label %[[AFTER_LT:.*]]
  //
  // LLVM: [[LT_FALSE]]:
  // LLVM:   br label %[[AFTER_LT]]
  //
  // LLVM: [[AFTER_LT]]:
  // LLVM:   %[[LT_RES_PHI:.*]] = phi ptr [ @_ZNSt3__115strong_ordering7greaterE, %[[LT_FALSE]] ], [ @_ZNSt3__115strong_ordering4lessE, %[[LT_TRUE]] ]
  // LLVM:   br label %[[AFTER_LT_CTD:.*]]
  //
  // LLVM: [[AFTER_LT_CTD]]:
  // LLVM:   br label %[[AFTER_CMPS:.*]]
  //
  // LLVM: [[AFTER_CMPS]]:
  // LLVM:   %[[CMP_RES:.*]] = phi ptr [ %[[LT_RES_PHI]], %[[AFTER_LT_CTD]] ], [ @_ZNSt3__115strong_ordering5equalE, %[[EQ_TRUE]] ]
  // LLVM:   br label %[[AFTER_CMPS_CTD:.*]]
  //
  // LLVM: [[AFTER_CMPS_CTD]]:
  // LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP_SO]], ptr %[[CMP_RES]], i64 1, i1 false)
  // LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[RET_ALLOCA]], ptr %[[TMP_SO]], i64 1, i1 false)
  // LLVM:   %[[RET_LOAD:.*]] = load %"class.std::__1::strong_ordering", ptr %[[RET_ALLOCA]]
  // LLVM:   %[[SO_NE_RES:.*]] = call noundef i1 @_ZNSt3__1neENS_15strong_orderingEMNS_19_CmpUnspecifiedTypeEFvvE(%"class.std::__1::strong_ordering" %[[RET_LOAD]],
  // LLVM:   br i1 %[[SO_NE_RES]], label %[[SO_NE_RES_TRUE:.*]], label %[[SO_NE_RES_FALSE:.*]]
  //
  // LLVM: [[SO_NE_RES_TRUE]]:
  // LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP_SO2]], ptr %[[TMP_SO]], i64 1, i1 false)
  // LLVM:   %[[TMP_SO2_LOAD:.*]] = load %"class.std::__1::strong_ordering", ptr %[[TMP_SO2]]
  // LLVM:   ret %"class.std::__1::strong_ordering" %[[TMP_SO2_LOAD]]
  //
  // LLVM: [[SO_NE_RES_FALSE]]:
  // LLVM:   br label %[[SO_NE_RES_FALSE_CTD:.*]]
  //
  // LLVM: [[SO_NE_RES_FALSE_CTD]]:
  // LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[TMP_SO2]], ptr @_ZNSt3__115strong_ordering5equalE, i64 1, i1 false)
  // LLVM:   %[[TMP_SO2_LOAD:.*]] = load %"class.std::__1::strong_ordering", ptr %[[TMP_SO2]]
  // LLVM:   ret %"class.std::__1::strong_ordering" %[[TMP_SO2_LOAD]]
  // LLVM: }

  // LLVM: define {{.*}}void @_Z19use_pseudo_ordering9HasMemberS_(%struct.HasMember %{{.*}}, %struct.HasMember %{{.*}}) #0 {
  // LLVM:   %[[M1_ALLOCA:.*]] = alloca %struct.HasMember
  // LLVM:   %[[M2_ALLOCA:.*]] = alloca %struct.HasMember
  // LLVM:   %[[G_ALLOCA:.*]] = alloca %"class.std::__1::strong_ordering"
  // LLVM:   %[[CALL_RES:.*]] = call %"class.std::__1::strong_ordering" @_ZNK9HasMemberssERKS_(ptr {{.*}}%[[M1_ALLOCA]], ptr {{.*}}%[[M2_ALLOCA]])
  // LLVM:   store %"class.std::__1::strong_ordering" %[[CALL_RES]], ptr %[[G_ALLOCA]]
  // LLVM:   ret void
  // LLVM: }

  // OGCG: define {{.*}}void @_Z19use_pseudo_ordering9HasMemberS_()
  // OGCG:   %[[M1_ALLOCA:.*]] = alloca %struct.HasMember
  // OGCG:   %[[M2_ALLOCA:.*]] = alloca %struct.HasMember
  // OGCG:   %[[G_ALLOCA:.*]] = alloca %"class.std::__1::strong_ordering"
  // OGCG:   %[[CALL_RES:.*]] = call i8 @_ZNK9HasMemberssERKS_(ptr {{.*}}%[[M1_ALLOCA]], ptr {{.*}}%[[M2_ALLOCA]])
  // OGCG:   %[[G_COERCE:.*]] = getelementptr inbounds nuw %"class.std::__1::strong_ordering", ptr %[[G_ALLOCA]], i32 0, i32 0
  // OGCG:   store i8 %[[CALL_RES]], ptr %[[G_COERCE]]
  // OGCG:   ret void
  // OGCG: }
  //
  // OGCG: define {{.*}}i8 @_ZNK9HasMemberssERKS_(ptr {{.*}}, ptr {{.*}})
  // OGCG:   %[[RET_ALLOCA:.*]] = alloca %"class.std::__1::strong_ordering"
  // OGCG:   %[[LHS_ALLOCA:.*]] = alloca ptr
  // OGCG:   %[[RHS_ALLOCA:.*]] = alloca ptr
  // OGCG:   %[[CMP_RES:.*]] = alloca %"class.std::__1::strong_ordering"
  // OGCG:   %[[CMP_TEMP:.*]] = alloca %"class.std::__1::strong_ordering"
  // OGCG:   %[[COERCE_ALLOCA:.*]] = alloca { i64, i64 }
  // OGCG:   %[[LHS_LOAD:.*]] = load ptr, ptr %[[LHS_ALLOCA]]
  // OGCG:   %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
  // OGCG:   %[[EQ_RES:.*]] = call {{.*}}i1 @_ZNK6MembereqERKS_(ptr {{.*}}%[[LHS_LOAD]], ptr {{.*}}%[[RHS_LOAD]])
  // OGCG:   br i1 %[[EQ_RES]], label %[[EQ_TRUE:.*]], label %[[EQ_FALSE:.*]]
  //
  // OGCG: [[EQ_TRUE]]:
  // OGCG:   br label %[[AFTER_COND:.*]]
  //
  // OGCG: [[EQ_FALSE]]:
  // OGCG:   %[[LT_RES:.*]] = call {{.*}}i1 @_ZNK6MemberltERKS_(ptr {{.*}}%[[LHS_LOAD]], ptr {{.*}}%[[RHS_LOAD]])
  // OGCG:   br i1 %[[LT_RES]], label %[[LT_TRUE:.*]], label %[[LT_FALSE:.*]]
  //
  // OGCG: [[LT_TRUE]]:
  // OGCG:   br label %[[AFTER_LT:.*]]
  //
  // OGCG: [[LT_FALSE]]:
  // OGCG:   br label %[[AFTER_LT]]
  // 
  // OGCG: [[AFTER_LT]]:
  // OGCG:   %[[AFTER_LT_RES:.*]] = phi ptr [ @_ZNSt3__115strong_ordering4lessE, %[[LT_TRUE]] ], [ @_ZNSt3__115strong_ordering7greaterE, %[[LT_FALSE]] ]
  // OGCG:   br label %cond.end5
  //
  // OGCG: [[AFTER_COND]]:
  // OGCG:   %[[MEM_EQ_RES:.*]] = phi ptr [ @_ZNSt3__115strong_ordering5equalE, %[[EQ_TRUE]] ], [ %[[AFTER_LT_RES]], %[[AFTER_LT]] ]
  // OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[CMP_RES]], ptr align 1 %[[MEM_EQ_RES]], i64 1, i1 false)
  // OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[CMP_TEMP]], ptr align 1 %[[CMP_RES]], i64 1, i1 false)
  // OGCG:   %[[UNSPEC_TEMP:.*]] = getelementptr inbounds nuw %"class.std::__1::strong_ordering", ptr %[[CMP_TEMP]], i32 0, i32 0
  // OGCG:   %[[UNSPEC_LOAD:.*]] = load i8, ptr %[[UNSPEC_TEMP]]
  // OGCG:   store { i64, i64 } zeroinitializer, ptr %[[COERCE_ALLOCA]]
  // OGCG:   %[[COERCE_L_GEP:.*]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[COERCE_ALLOCA]], i32 0, i32 0
  // OGCG:   %[[COERCE_L:.*]] = load i64, ptr %[[COERCE_L_GEP]]
  // OGCG:   %[[COERCE_R_GEP:.*]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[COERCE_ALLOCA]], i32 0, i32 1
  // OGCG:   %[[COERCE_R:.*]] = load i64, ptr %[[COERCE_R_GEP]]
  // OGCG:   %[[UNSPEC_RES:.*]] = call noundef zeroext i1 @_ZNSt3__1neENS_15strong_orderingEMNS_19_CmpUnspecifiedTypeEFvvE(i8 %[[UNSPEC_LOAD]], i64 %[[COERCE_L]], i64 %[[COERCE_R]])
  // OGCG:   br i1 %[[UNSPEC_RES]], label %[[UNSPEC_TRUE:.*]], label %[[UNSPEC_FALSE:.*]]
  //
  // OGCG: [[UNSPEC_TRUE]]:
  // OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[RET_ALLOCA]], ptr align 1 %[[CMP_RES]], i64 1, i1 false)
  // OGCG:   br label %[[RETURN_BLOCK:.*]]
  // 
  // OGCG: [[UNSPEC_FALSE]]:
  // OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[RET_ALLOCA]], ptr align 1 @_ZNSt3__115strong_ordering5equalE, i64 1, i1 false)
  // OGCG:   br label %[[RETURN_BLOCK]]
  //
  // OGCG: [[RETURN_BLOCK]]:
  // OGCG:   %[[GEP_RET:.*]] = getelementptr inbounds nuw %"class.std::__1::strong_ordering", ptr %[[RET_ALLOCA]], i32 0, i32 0
  // OGCG:   %[[GEP_RET_LOAD:.*]] = load i8, ptr %[[GEP_RET]]
  // OGCG:   ret i8 %[[GEP_RET_LOAD]]
}
