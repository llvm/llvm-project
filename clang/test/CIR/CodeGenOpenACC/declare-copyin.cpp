// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct HasSideEffects {
  HasSideEffects();
  ~HasSideEffects();
};

HasSideEffects GlobalHSE1;
HasSideEffects GlobalHSEArr[5];
int GlobalInt1;

#pragma acc declare copyin(GlobalHSE1, GlobalInt1, GlobalHSEArr[1:1])
// CHECK: acc.global_ctor @GlobalHSE1_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalHSE1 : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "GlobalHSE1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @GlobalHSE1_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalHSE1 : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copyin>, name = "GlobalHSE1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, name = "GlobalHSE1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @GlobalInt1_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalInt1 : !cir.ptr<!s32i>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "GlobalInt1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @GlobalInt1_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalInt1 : !cir.ptr<!s32i>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyin>, name = "GlobalInt1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "GlobalInt1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @GlobalHSEArr_acc_ctor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalHSEArr : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "GlobalHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @GlobalHSEArr_acc_dtor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @GlobalHSEArr : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copyin>, name = "GlobalHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copyin>, name = "GlobalHSEArr[1:1]"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }

namespace NS {

HasSideEffects NSHSE1;
HasSideEffects NSHSEArr[5];
int NSInt1;

#pragma acc declare copyin(alwaysin: NSHSE1, NSInt1, NSHSEArr[1:1])
// CHECK: acc.global_ctor @{{.*}}NSHSE1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSE1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}NSHSE1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSE1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSE1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}NSInt1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier alwaysin>, name = "NSInt1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}NSInt1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSInt1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSInt1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}NSHSEArr{{.*}}_acc_ctor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}NSHSEArr{{.*}}_acc_dtor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}NSHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "NSHSEArr[1:1]"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }

} // namespace NS

namespace {

HasSideEffects AnonNSHSE1;
HasSideEffects AnonNSHSEArr[5];
int AnonNSInt1;

#pragma acc declare copyin(AnonNSHSE1, AnonNSInt1, AnonNSHSEArr[1:1])
// CHECK: acc.global_ctor @{{.*}}AnonNSHSE1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "AnonNSHSE1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}AnonNSHSE1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSHSE1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSHSE1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}AnonNSInt1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "AnonNSInt1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}AnonNSInt1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSInt1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSInt1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}AnonNSHSEArr{{.*}}_acc_ctor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "AnonNSHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}AnonNSHSEArr{{.*}}_acc_dtor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}AnonNSHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copyin>, name = "AnonNSHSEArr[1:1]"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }

} // namespace NS

struct Struct {
  static const HasSideEffects StaticMemHSE1;
  static const HasSideEffects StaticMemHSEArr[5];
  static const int StaticMemInt1;

#pragma acc declare copyin(StaticMemHSE1, StaticMemInt1, StaticMemHSEArr[1:1])
// CHECK: acc.global_ctor @{{.*}}Struct{{.*}}StaticMemHSE1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}{{.*}}Struct{{.*}}StaticMemHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "StaticMemHSE1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}Struct{{.*}}StaticMemHSE1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}Struct{{.*}}StaticMemHSE1{{.*}} : !cir.ptr<!rec_HasSideEffects>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemHSE1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemHSE1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}Struct{{.*}}StaticMemInt1{{.*}}_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}Struct{{.*}}StaticMemInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "StaticMemInt1"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}Struct{{.*}}StaticMemInt1{{.*}}_acc_dtor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}Struct{{.*}}StaticMemInt1{{.*}} : !cir.ptr<!s32i>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemInt1"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemInt1"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
//
// CHECK: acc.global_ctor @{{.*}}Struct{{.*}}StaticMemHSEArr{{.*}}_acc_ctor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}Struct{{.*}}StaticMemHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "StaticMemHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }
// CHECK: acc.global_dtor @{{.*}}Struct{{.*}}StaticMemHSEArr{{.*}}_acc_dtor {
// CHECK-NEXT: %[[LB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]]
// CHECK-NEXT: %[[UB:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]]
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[LB_CAST]] : si32) extent(%[[UB_CAST]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @{{.*}}Struct{{.*}}StaticMemHSEArr{{.*}} : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>
// CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[GET_GLOBAL]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemHSEArr[1:1]"}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
// CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) {dataClause = #acc<data_clause acc_copyin>, name = "StaticMemHSEArr[1:1]"}
// CHECK-NEXT: acc.terminator
// CHECK-NEXT: }

  void MemFunc1(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc1{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load

    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    int LocalInt;

#pragma acc declare copyin(always:ArgHSE, ArgInt, LocalHSE, LocalInt, ArgHSEPtr[1:1], LocalHSEArr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always>, name = "ArgInt"} 
    // CHECK-NEXT: %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier always>, name = "LocalHSE"} 
    // CHECK-NEXT: %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always>, name = "LocalInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {modifiers = #acc<data_clause_modifier always>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT: %[[ENTER:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    //
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: } cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgInt"}
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSE"}
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "LocalInt"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
  }
  void MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr);
};

void use() {
  Struct s;
  s.MemFunc1(HasSideEffects{}, 0, nullptr);
}

void Struct::MemFunc2(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}MemFunc2{{.*}}(%{{.*}}: !cir.ptr<!rec_Struct>{{.*}}, %[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: cir.alloca{{.*}}["this"
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.load
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copyin(alwaysin:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgInt"} 
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)

#pragma acc declare copyin(alwaysin:LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalHSE"} 
    // CHECK-NEXT:   %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalInt"}
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT:   %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT:   %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT:   %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT:   %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT:   %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT:   %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    // CHECK-NEXT:   cir.cleanup.scope {
    // CHECK-NEXT:     cir.yield
    // CHECK-NEXT:   } cleanup normal {
    // CHECK-NEXT:     acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT:     acc.delete accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalHSE"}
    // CHECK-NEXT:     acc.delete accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalInt"}
    // CHECK-NEXT:     acc.delete accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT:     cir.yield
    // CHECK-NEXT:  }

    // CHECK-NEXT:   cir.yield

    // CHECK-NEXT: } cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSE"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgInt"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier alwaysin>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
}

extern "C" void do_thing();

extern "C" void NormalFunc(HasSideEffects ArgHSE, int ArgInt, HasSideEffects *ArgHSEPtr) {
    // CHECK: cir.func {{.*}}NormalFunc(%[[ARG_HSE:.*]]: !rec_HasSideEffects{{.*}}, %[[ARG_INT:.*]]: !s32i {{.*}}, %[[ARG_HSE_PTR:.*]]: !cir.ptr<!rec_HasSideEffects>{{.*}})
    // CHECK-NEXT: %[[ARG_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["ArgHSE"
    // CHECK-NEXT: %[[ARG_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["ArgInt
    // CHECK-NEXT: %[[ARG_HSE_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasSideEffects>{{.*}}["ArgHSEPtr"
    // CHECK-NEXT: %[[LOC_HSE_ALLOCA:.*]] = cir.alloca !rec_HasSideEffects{{.*}}["LocalHSE
    // CHECK-NEXT: %[[LOC_HSE_ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_HasSideEffects x 5>{{.*}}["LocalHSEArr
    // CHECK-NEXT: %[[LOC_INT_ALLOCA:.*]] = cir.alloca !s32i{{.*}}["LocalInt
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    // CHECK-NEXT: cir.store
    HasSideEffects LocalHSE;
    // CHECK-NEXT: cir.call{{.*}} : (!cir.ptr<!rec_HasSideEffects>) -> ()
    HasSideEffects LocalHSEArr[5];
    // CHECK: do {
    // CHECK: } while {
    // CHECK: }
    int LocalInt;
#pragma acc declare copyin(always:ArgHSE, ArgInt, ArgHSEPtr[1:1])
    // CHECK: %[[ARG_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT: %[[ARG_INT_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always>, name = "ArgInt"} 
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[ARG_HSE_PTR_COPYIN:.*]] = acc.copyin varPtr(%[[ARG_HSE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) -> !cir.ptr<!cir.ptr<!rec_HasSideEffects>> {modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
    // CHECK-NEXT: %[[ENTER1:.*]] = acc.declare_enter dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    {
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT: cir.scope {
#pragma acc declare copyin(LocalHSE, LocalInt, LocalHSEArr[1:1])
    // CHECK-NEXT: %[[LOC_HSE_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ALLOCA]] : !cir.ptr<!rec_HasSideEffects>) -> !cir.ptr<!rec_HasSideEffects> {name = "LocalHSE"} 
    // CHECK-NEXT: %[[LOC_INT_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_INT_ALLOCA]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "LocalInt"}
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[LB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
    // CHECK-NEXT: %[[UB:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
    // CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[STRIDE:.*]] = arith.constant 1 : i64
    // CHECK-NEXT: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : si32) extent(%[[UB]] : si32) stride(%[[STRIDE]] : i64) startIdx(%[[IDX]] : i64)
    // CHECK-NEXT: %[[LOC_HSE_ARR_COPYIN:.*]] = acc.copyin varPtr(%[[LOC_HSE_ARR_ALLOCA]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) -> !cir.ptr<!cir.array<!rec_HasSideEffects x 5>> {name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT: %[[ENTER2:.*]] = acc.declare_enter dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)

    do_thing();
    // CHECK-NEXT: cir.cleanup.scope {
    // CHECK-NEXT:   cir.call @do_thing
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER2]]) dataOperands(%[[LOC_HSE_COPYIN]], %[[LOC_INT_COPYIN]], %[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>)
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, name = "LocalHSE"}
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "LocalInt"}
    // CHECK-NEXT:   acc.delete accPtr(%[[LOC_HSE_ARR_COPYIN]] : !cir.ptr<!cir.array<!rec_HasSideEffects x 5>>) bounds(%[[BOUND2]]) {dataClause = #acc<data_clause acc_copyin>, name = "LocalHSEArr[1:1]"}
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: }
    }
    // CHECK-NEXT: }

    // Make sure that cleanup gets put in the right scope.
    do_thing();
    // CHECK-NEXT:   cir.call @do_thing
    // CHECK-NEXT:   cir.yield
    // CHECK-NEXT: cleanup normal {
    // CHECK-NEXT:   acc.declare_exit token(%[[ENTER1]]) dataOperands(%[[ARG_HSE_COPYIN]], %[[ARG_INT_COPYIN]], %[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!rec_HasSideEffects>, !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!rec_HasSideEffects>>)
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_COPYIN]] : !cir.ptr<!rec_HasSideEffects>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSE"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_INT_COPYIN]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgInt"}
    // CHECK-NEXT:   acc.delete accPtr(%[[ARG_HSE_PTR_COPYIN]] : !cir.ptr<!cir.ptr<!rec_HasSideEffects>>) bounds(%[[BOUND1]]) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "ArgHSEPtr[1:1]"}
}

