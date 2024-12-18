// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {

  func.func @_QQmain() attributes {fir.bindc_name = "cufkernel_global"} {
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@_QQclX3C737464696E3E00) : !fir.ref<!fir.char<1,8>>
    %c4_i32 = arith.constant 4 : i32
    %c48 = arith.constant 48 : index
    %1 = fir.convert %c48 : (index) -> i64
    %2 = fir.convert %0 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
    %3 = fir.call @_FortranACUFAllocDesciptor(%1, %2, %c4_i32) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
    %4 = fir.convert %3 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %5 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
    %6 = fircg.ext_embox %5(%c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %6 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %8 = fir.load %3 : !fir.ref<!fir.box<none>>
    return
  }

  // CHECK-LABEL: llvm.func @_QQmain()
  // CHECK-COUNT-2: llvm.call @_FortranACUFAllocDesciptor 

  fir.global linkonce @_QQclX3C737464696E3E00 constant : !fir.char<1,8> {
    %0 = fir.string_lit "<stdin>\00"(8) : !fir.char<1,8>
    fir.has_value %0 : !fir.char<1,8>
  }
  func.func private @_FortranACUFAllocDesciptor(i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>> attributes {fir.runtime}
}
