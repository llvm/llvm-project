// RUN: inter-opt --split-input-file -verify-diagnostics %s

// expected-error@+1 {{SIMD cardinality must be positive}}
func.func @bad_type(%arg: !xw.simd<i32, 0>)

// -----

// expected-error@+1 {{'xw.simd_width' must be 8, 16, or 32}}
func.func @bad_width() attributes {xw.simd_width = 64 : i64}

// -----

// expected-error@+1 {{does not divide xw.simd_width 32}}
func.func @bad_argument(%arg: !xw.simd<i32, 12>)
    attributes {xw.simd_width = 32 : i64}

// -----

func.func @missing_width(%arg: i32) {
  // expected-error@+1 {{requires an enclosing xw.simd_width}}
  %value = xw.splat %arg : i32 -> !xw.simd<i32, 8>
  return
}

// -----

func.func @binary_mismatch(%a: !xw.simd<i32, 8>,
                           %b: !xw.simd<i32, 16>)
    attributes {xw.simd_width = 32 : i64} {
  // expected-error@+1 {{use xw.expand explicitly}}
  %value = xw.binary addi %a, %b
      : !xw.simd<i32, 8>, !xw.simd<i32, 16> -> !xw.simd<i32, 16>
  return
}

// -----

func.func @bad_cast_policy(%arg: i32)
    attributes {xw.simd_width = 32 : i64} {
  // expected-error@+1 {{signedness policy is required}}
  %value = xw.cast int_to_fp %arg : i32 -> f32
  return
}

// -----

func.func @bad_pointer_predicate(%a: !xw.ptr<#xw.global>,
                                 %b: !xw.ptr<#xw.global>) {
  // expected-error@+1 {{predicate must be eq or ne}}
  %value = xw.ptr_cmp ult %a, %b
      : !xw.ptr<#xw.global>, !xw.ptr<#xw.global> -> i1
  return
}

// -----

func.func @bad_local_base() {
  // expected-error@+1 {{result must use the local address space}}
  %base = xw.local_memory_base : !xw.ptr<#xw.global>
  return
}

// -----

func.func @freeze_token(%token: !xw.mem.token) {
  // expected-error@+1 {{requires a bare or SIMD signless integer, index, floating-point, or XW pointer payload}}
  %frozen = xw.freeze %token : !xw.mem.token
  return
}

// -----

func.func @freeze_aggregate(%value: vector<2xi32>) {
  // expected-error@+1 {{requires a bare or SIMD signless integer, index, floating-point, or XW pointer payload}}
  %frozen = xw.freeze %value : vector<2xi32>
  return
}

// -----

func.func @freeze_missing_width(%value: !xw.simd<i32, 8>) {
  // expected-error@+1 {{requires an enclosing xw.simd_width}}
  %frozen = xw.freeze %value : !xw.simd<i32, 8>
  return
}

// -----

func.func @overflow_xor(%a: i32, %b: i32) {
  // expected-error@+1 {{overflow flags require addi, subi, muli, or shli operation}}
  %value = xw.binary xori %a, %b overflow<nuw> : i32, i32 -> i32
  return
}

// -----

func.func @overflow_widen(%value: i32) {
  // expected-error@+1 {{overflow flags require a narrowing intconvert operation}}
  %wide = xw.cast intconvert %value overflow<nsw> : i32 -> i64
  return
}

// -----

func.func @multiblock_vnni(%base: !xw.ptr<#xw.global>, %geometry: i32)
    attributes {xw.simd_width = 16 : i32} {
  // expected-error@+1 {{transformed block2D reads require one block}}
  %value, %token = xw.block2d_read %base[%geometry, %geometry]
      surface (%geometry, %geometry, %geometry)
      {block_height = 16 : i64, block_width = 16 : i64, blocks = 2 : i64,
       element_bits = 16 : i64, vnni}
      : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32)
        -> (!xw.simd<vector<16xi16>, 16>, !xw.mem.token)
  return
}
