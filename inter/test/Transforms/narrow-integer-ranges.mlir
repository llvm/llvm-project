// RUN: inter-opt %s --inter-narrow-integer-ranges --canonicalize | FileCheck %s

module {
  func.func @narrow_block2d_loop(%base: !xw.ptr<#xw.global>) {
    %c0 = xw.constant 0 : i64
    %c31 = xw.constant 31 : i64
    %c32 = xw.constant 32 : i64
    %c4096 = xw.constant 4096 : i64
    %surface = xw.constant 4096 : i32
    %subgroup = xw.subgroup_id : i32
    %wide = xw.cast intconvert %subgroup policy {extension = #xw.cast_extension<zero>} : i32 -> i64
    %offset = xw.binary andi %wide, %c31 : i64, i64 -> i64
    scf.for %iv = %c0 to %c4096 step %c32 : i64 {
      %next = xw.binary addi %iv, %c32 : i64, i64 -> i64
      %coordinate = xw.binary addi %offset, %next : i64, i64 -> i64
      %x = xw.cast intconvert %coordinate : i64 -> i32
      %token = xw.block2d_prefetch %base[%x, %x]
          surface(%surface, %surface, %surface)
          {block_height = 8 : i64, block_width = 16 : i64,
           blocks = 1 : i64, element_bits = 16 : i64}
          : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32) -> !xw.mem.token
    }
    return
  }
}

// CHECK-LABEL: func.func @narrow_block2d_loop
// CHECK: scf.for %[[IV:.*]] = {{.*}} : i32
// CHECK: %[[NEXT:.*]] = xw.binary addi %[[IV]], {{.*}} : i32, i32 -> i32
// CHECK: %[[COORD:.*]] = xw.binary addi {{.*}}, %[[NEXT]] : i32, i32 -> i32
// CHECK: xw.block2d_prefetch {{.*}}[%[[COORD]], %[[COORD]]]
// CHECK-NOT: xw.cast intconvert %[[COORD]]

// -----

module {
  func.func @preserve_dynamic_bound(%base: !xw.ptr<#xw.global>,
                                    %upper: i64, %offset: i64) {
    %c0 = xw.constant 0 : i64
    %c32 = xw.constant 32 : i64
    %surface = xw.constant 4096 : i32
    scf.for %iv = %c0 to %upper step %c32 : i64 {
      %coordinate = xw.binary addi %offset, %iv : i64, i64 -> i64
      %x = xw.cast intconvert %coordinate : i64 -> i32
      %token = xw.block2d_prefetch %base[%x, %x]
          surface(%surface, %surface, %surface)
          {block_height = 8 : i64, block_width = 16 : i64,
           blocks = 1 : i64, element_bits = 16 : i64}
          : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32) -> !xw.mem.token
    }
    return
  }
}

// CHECK-LABEL: func.func @preserve_dynamic_bound
// CHECK: scf.for %[[IV:.*]] = {{.*}} : i64
// CHECK: xw.binary addi {{.*}}, %[[IV]] : i64, i64 -> i64

// -----

module {
  func.func @preserve_overflowing_bound(%base: !xw.ptr<#xw.global>) {
    %c0 = xw.constant 0 : i64
    %c32 = xw.constant 32 : i64
    %limit = xw.constant 2147483647 : i64
    %surface = xw.constant 4096 : i32
    scf.for %iv = %c0 to %limit step %c32 : i64 {
      %x = xw.cast intconvert %iv : i64 -> i32
      %token = xw.block2d_prefetch %base[%x, %x]
          surface(%surface, %surface, %surface)
          {block_height = 8 : i64, block_width = 16 : i64,
           blocks = 1 : i64, element_bits = 16 : i64}
          : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32) -> !xw.mem.token
    }
    return
  }
}

// CHECK-LABEL: func.func @preserve_overflowing_bound
// CHECK: scf.for %{{.*}} = {{.*}} : i64

// -----

module {
  func.func @preserve_wide_shift_amount(%arg: i64) -> i64 {
    %zero = xw.constant 0 : i64
    %mask = xw.constant 63 : i64
    %amount = xw.binary andi %arg, %mask : i64, i64 -> i64
    %result = xw.binary shli %zero, %amount : i64, i64 -> i64
    return %result : i64
  }
}

// CHECK-LABEL: func.func @preserve_wide_shift_amount
// CHECK: xw.binary shli {{.*}} : i64, i64 -> i64

// -----

module {
  func.func @preserve_unsigned_semantics(%arg: i64) -> i64 {
    %zero = xw.constant 0 : i64
    %one = xw.constant 1 : i64
    %seven = xw.constant 7 : i64
    %bit = xw.binary andi %arg, %one : i64, i64 -> i64
    %negative = xw.binary subi %zero, %bit : i64, i64 -> i64
    %result = xw.binary remui %negative, %seven : i64, i64 -> i64
    return %result : i64
  }
}

// CHECK-LABEL: func.func @preserve_unsigned_semantics
// CHECK: xw.binary remui {{.*}} : i64, i64 -> i64
