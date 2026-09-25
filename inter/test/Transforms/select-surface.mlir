// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @surface(%global: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 8 : i32} {
    %zero = xw.constant 0 : i32
    %one = xw.constant 1 : i32
    %lanes = xw.splat %one : i32 -> !xw.simd<i32, 8>
    %xor = xw.binary xori %lanes, %zero
        : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
    %eq = xw.cmpi eq %lanes, %xor
        : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.mask<8>
    %ne = xw.cmpi ne %lanes, %xor
        : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.mask<8>
    %and = xw.mask_and %eq, %ne : !xw.mask<8>
    %or = xw.mask_or %eq, %ne : !xw.mask<8>
    %mask_xor = xw.mask_xor %and, %or : !xw.mask<8>
    %not = xw.mask_not %mask_xor : !xw.mask<8>
    %ballot = xw.ballot %not : !xw.mask<8> -> i8
    %pointer_bits = xw.ptr_to_int %global : !xw.ptr<#xw.global> -> i64
    %roundtrip = xw.int_to_ptr %pointer_bits
        : i64 -> !xw.ptr<#xw.global>
    %generic = xw.addrspace_cast %roundtrip
        : !xw.ptr<#xw.global> -> !xw.ptr<#xw.generic>
    %null = xw.null : !xw.ptr<#xw.global>
    %is_null = xw.ptr_cmp eq %global, %null
        : !xw.ptr<#xw.global>, !xw.ptr<#xw.global> -> i1
    %lane = xw.constant 3 : i32
    %shuffle = xw.shuffle %xor from %lane
        : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
    %block = xw.launch_block_size 1 : i32
    %allocation = xw.alloc() {bytesize = 32 : i64, align = 16 : i64}
        : !xw.ptr<#xw.local>
    %local_bits = xw.ptr_to_int %allocation
        : !xw.ptr<#xw.local> -> i32
    %local = xw.int_to_ptr %local_bits : i32 -> !xw.ptr<#xw.local>
    %root = xw.token : !xw.mem.token
    %released = xw.alloc_release %local after %root
        : (!xw.ptr<#xw.local>, !xw.mem.token) -> !xw.mem.token
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @surface
// CHECK: xemachine.or
// CHECK: xemachine.and
// CHECK: xemachine.sub
// CHECK: xemachine.cmp
// CHECK: xemachine.mov
// CHECK: src0Sub = 3 : i32} : {{.*}}i64
// CHECK: src0Sub = 4
// CHECK: xemachine.token
// CHECK: xemachine.eot
