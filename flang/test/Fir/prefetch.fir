// Test lowering of prefetch directive from FIR to LLVM IR
// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

func.func @_QPtest(%arg0: !fir.ref<i32>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>) {
    // CHECK: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
    fir.prefetch %arg0#0 {read, data, localityHint = 3 : i32} : !fir.ref<i32>
    %c2 = arith.constant 2 : index
    %4 = hlfir.designate %arg1#0 (%c2)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>

    // CHECK: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
    fir.prefetch %4 {read, data, localityHint = 3 : i32} : !fir.ref<f32>
    %5 = fir.load %arg2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>
    %6 = fir.box_addr %5 : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,4>>>

    // CHECK: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
    fir.prefetch %6 {read, data, localityHint = 3 : i32} : !fir.heap<!fir.array<?x!fir.char<1,4>>>
    %7 = fir.load %arg2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %8 = hlfir.designate %7 (%c3)  typeparams %c4 : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>, index, index) -> !fir.ref<!fir.char<1,4>>

    // CHECK: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
    fir.prefetch %8 {read, data, localityHint = 3 : i32} : !fir.ref<!fir.char<1,4>>
    return
}
