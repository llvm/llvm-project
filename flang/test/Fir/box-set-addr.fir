// Test fir.box_set_addr parsing/printing and conversion to LLVM.
// RUN: fir-opt %s | fir-opt | FileCheck %s --check-prefix=FIR
// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu apply-tbaa=true" --per-function-tbaa-trees=false %s | FileCheck %s --check-prefix=LLVM

// LLVM-DAG:     #[[ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root ">
// LLVM-DAG:     #[[ANYACC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// LLVM-DAG:     #[[BOXMEM:.*]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#[[ANYACC]], 0>}>
// LLVM-DAG:     #[[$BOXT:.*]] = #llvm.tbaa_tag<base_type = #[[BOXMEM]], access_type = #[[BOXMEM]], offset = 0>

func.func @set_addr_array(%p : !fir.ptr<i64>, %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
  fir.box_set_addr %p to %desc : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  return
}
// FIR-LABEL:   func.func @set_addr_array(
// FIR-SAME:      %[[P:.*]]: !fir.ptr<i64>, %[[D:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// FIR:           fir.box_set_addr %[[P]] to %[[D]] : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

// LLVM-LABEL:   llvm.func @set_addr_array(
// LLVM-SAME:      %[[P:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr)
// LLVM:           %[[F:.*]] = llvm.getelementptr %[[D]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// LLVM-NEXT:      llvm.store %[[P]], %[[F]] <tbaa = [#[[$BOXT]]]> : !llvm.ptr, !llvm.ptr
// LLVM-NEXT:      llvm.return

func.func @set_addr_char(%p : !fir.ptr<i64>, %desc : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) {
  fir.box_set_addr %p to %desc : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  return
}
// FIR-LABEL:   func.func @set_addr_char(
// FIR:           fir.box_set_addr %{{.*}} to %{{.*}} : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>

// LLVM-LABEL:   llvm.func @set_addr_char(
// LLVM-SAME:      %[[P:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr)
// LLVM:           %[[F:.*]] = llvm.getelementptr %[[D]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
// LLVM-NEXT:      llvm.store %[[P]], %[[F]] <tbaa = [#[[$BOXT]]]> : !llvm.ptr, !llvm.ptr

func.func @set_addr_derived(%p : !fir.ptr<i64>, %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t{r:f32,i:i32}>>>>>) {
  fir.box_set_addr %p to %desc : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t{r:f32,i:i32}>>>>>
  return
}
// FIR-LABEL:   func.func @set_addr_derived(
// FIR:           fir.box_set_addr %{{.*}} to %{{.*}} : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t{r:f32,i:i32}>>>>>

// LLVM-LABEL:   llvm.func @set_addr_derived(
// LLVM-SAME:      %[[P:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr)
// LLVM:           %[[F:.*]] = llvm.getelementptr %[[D]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr, array<1 x i64>)>
// LLVM-NEXT:      llvm.store %[[P]], %[[F]] <tbaa = [#[[$BOXT]]]> : !llvm.ptr, !llvm.ptr

func.func @set_addr_volatile(%p : !fir.ptr<i64>, %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>, volatile>, volatile>) {
  fir.box_set_addr %p to %desc : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>, volatile>, volatile>
  return
}
// FIR-LABEL:   func.func @set_addr_volatile(
// FIR:           fir.box_set_addr %{{.*}} to %{{.*}} : !fir.ptr<i64>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>, volatile>, volatile>

// LLVM-LABEL:   llvm.func @set_addr_volatile(
// LLVM-SAME:      %[[P:.*]]: !llvm.ptr, %[[D:.*]]: !llvm.ptr)
// LLVM:           %[[F:.*]] = llvm.getelementptr %[[D]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
// LLVM-NEXT:      llvm.store volatile %[[P]], %[[F]] <tbaa = [#[[$BOXT]]]> : !llvm.ptr, !llvm.ptr
