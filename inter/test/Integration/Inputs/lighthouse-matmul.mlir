module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.module_asm = [], llvm.target_triple = ""} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>]
  llvm.func spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {no_unwind, will_return}
  llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
  llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
  llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
  llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind}
  llvm.func spir_funccc @_Z16get_sub_group_id() -> i32 attributes {no_unwind, will_return}
  llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
  func.func @payload_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<1>, %arg15: !llvm.ptr<1>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {xw.kernel, xw.kernel_args = [{access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 24 : i64, size = 8 : i64}, {access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 32 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 40 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 48 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 56 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 64 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 72 : i64, size = 8 : i64}, {access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 80 : i64, size = 8 : i64}, {access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 88 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 96 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 104 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 112 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 120 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 128 : i64, size = 8 : i64}, {access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 136 : i64, size = 8 : i64}, {access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 144 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 152 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 160 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 168 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 176 : i64, size = 8 : i64}, {alignment = 4 : i64, kind = "value", offset = 184 : i64, size = 8 : i64}], xw.required_work_group_size = [256 : i32, 1 : i32, 1 : i32], xw.simd_width = 16 : i32} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(dense<0> : vector<4xi64>) : vector<4xi64>
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(128 : i32) : i32
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(4 : i64) : i64
    %10 = llvm.mlir.constant(8 : i64) : i64
    %11 = llvm.mlir.constant(16 : i64) : i64
    %12 = llvm.mlir.constant(32 : i64) : i64
    %13 = llvm.mlir.undef : vector<2xi32>
    %14 = llvm.mlir.zero : !llvm.ptr<1>
    %15 = llvm.mlir.constant(256 : i32) : i32
    %16 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %17 = llvm.mlir.constant(8 : i32) : i32
    %18 = llvm.mlir.constant(512 : i32) : i32
    %19 = llvm.ptrtoint %arg15 : !llvm.ptr<1> to i64
    %20 = llvm.ptrtoint %arg8 : !llvm.ptr<1> to i64
    %21 = llvm.ptrtoint %arg1 : !llvm.ptr<1> to i64
    %22 = llvm.call spir_funccc @_Z12get_group_idj(%0) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (i32) -> i64
    %23 = llvm.call spir_funccc @_Z12get_group_idj(%1) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (i32) -> i64
    %24 = llvm.mul %22, %2 overflow<nsw> : i64
    %25 = llvm.mul %23, %2 overflow<nsw> : i64
    %26 = llvm.insertelement %21, %4[%3 : i64] : vector<4xi64>
    %27 = llvm.bitcast %26 : vector<4xi64> to vector<8xi32>
    %28 = llvm.insertelement %5, %27[%6 : i64] : vector<8xi32>
    %29 = llvm.insertelement %7, %28[%8 : i64] : vector<8xi32>
    %30 = llvm.insertelement %5, %29[%9 : i64] : vector<8xi32>
    %31 = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    %32 = llvm.zext %31 : i32 to i64
    %33 = llvm.urem %32, %6 : i64
    %34 = llvm.udiv %32, %6 : i64
    %35 = llvm.urem %34, %10 : i64
    %36 = llvm.mul %35, %10 : i64
    %37 = llvm.mul %33, %11 : i64
    %38 = llvm.urem %36, %2 : i64
    %39 = llvm.urem %37, %12 : i64
    %40 = llvm.add %38, %24 : i64
    %41 = llvm.bitcast %30 : vector<8xi32> to vector<4xi64>
    %42 = llvm.extractelement %41[%3 : i64] : vector<4xi64>
    %43 = llvm.trunc %39 : i64 to i32
    %44 = llvm.trunc %40 : i64 to i32
    %45 = llvm.inttoptr %42 : i64 to !llvm.ptr<1>
    %46 = llvm.insertelement %43, %13[%0 : i32] : vector<2xi32>
    %47 = llvm.insertelement %44, %46[%1 : i32] : vector<2xi32>
    llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%45, %7, %7, %7, %47) {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}} : (!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) -> ()
    %48 = llvm.insertelement %20, %4[%3 : i64] : vector<4xi64>
    %49 = llvm.bitcast %48 : vector<4xi64> to vector<8xi32>
    %50 = llvm.insertelement %7, %49[%6 : i64] : vector<8xi32>
    %51 = llvm.insertelement %5, %50[%8 : i64] : vector<8xi32>
    %52 = llvm.insertelement %7, %51[%9 : i64] : vector<8xi32>
    %53 = llvm.urem %32, %9 : i64
    %54 = llvm.udiv %32, %9 : i64
    %55 = llvm.urem %54, %9 : i64
    %56 = llvm.mul %55, %10 : i64
    %57 = llvm.mul %53, %11 : i64
    %58 = llvm.urem %56, %12 : i64
    %59 = llvm.urem %57, %2 : i64
    %60 = llvm.add %59, %25 : i64
    %61 = llvm.bitcast %52 : vector<8xi32> to vector<4xi64>
    %62 = llvm.extractelement %61[%3 : i64] : vector<4xi64>
    %63 = llvm.trunc %60 : i64 to i32
    %64 = llvm.trunc %58 : i64 to i32
    %65 = llvm.inttoptr %62 : i64 to !llvm.ptr<1>
    %66 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %67 = llvm.insertelement %64, %66[%1 : i32] : vector<2xi32>
    llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%65, %15, %5, %15, %67) {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}} : (!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) -> ()
    %68 = llvm.mul %55, %11 : i64
    %69 = llvm.mul %53, %12 : i64
    %70 = llvm.urem %68, %2 : i64
    %71 = llvm.urem %69, %12 : i64
    %72 = llvm.add %70, %24 : i64
    %73 = llvm.add %72, %10 : i64
    %74 = llvm.mul %55, %12 : i64
    %75 = llvm.urem %74, %12 : i64
    cf.br ^bb1(%3, %16, %16 : i64, vector<8xf32>, vector<8xf32>)
  ^bb1(%76: i64, %77: vector<8xf32>, %78: vector<8xf32>):  // 2 preds: ^bb0, ^bb2
    %79 = llvm.icmp "slt" %76, %2 : i64
    cf.cond_br %79, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %80 = llvm.add %76, %12 : i64
    %81 = llvm.add %58, %80 : i64
    %82 = llvm.trunc %81 : i64 to i32
    %83 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %84 = llvm.insertelement %82, %83[%1 : i32] : vector<2xi32>
    llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%65, %15, %5, %15, %84) {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}} : (!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) -> ()
    %85 = llvm.add %39, %80 : i64
    %86 = llvm.trunc %85 : i64 to i32
    %87 = llvm.insertelement %86, %13[%0 : i32] : vector<2xi32>
    %88 = llvm.insertelement %44, %87[%1 : i32] : vector<2xi32>
    llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%45, %7, %7, %7, %88) {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}} : (!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) -> ()
    %89 = llvm.add %71, %76 : i64
    %90 = llvm.trunc %89 : i64 to i32
    %91 = llvm.trunc %72 : i64 to i32
    %92 = llvm.insertelement %90, %13[%0 : i32] : vector<2xi32>
    %93 = llvm.insertelement %91, %92[%1 : i32] : vector<2xi32>
    %94 = llvm.alloca %17 x i16 {alignment = 2 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%45, %7, %7, %7, %93, %94) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %95 = llvm.load %94 {alignment = 16 : i64} : !llvm.ptr -> vector<8xi16>
    %96 = llvm.bitcast %95 : vector<8xi16> to vector<8xf16>
    %97 = llvm.add %89, %11 : i64
    %98 = llvm.trunc %97 : i64 to i32
    %99 = llvm.insertelement %98, %13[%0 : i32] : vector<2xi32>
    %100 = llvm.insertelement %91, %99[%1 : i32] : vector<2xi32>
    %101 = llvm.alloca %17 x i16 {alignment = 2 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%45, %7, %7, %7, %100, %101) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %102 = llvm.load %101 {alignment = 16 : i64} : !llvm.ptr -> vector<8xi16>
    %103 = llvm.bitcast %102 : vector<8xi16> to vector<8xf16>
    %104 = llvm.trunc %73 : i64 to i32
    %105 = llvm.insertelement %90, %13[%0 : i32] : vector<2xi32>
    %106 = llvm.insertelement %104, %105[%1 : i32] : vector<2xi32>
    %107 = llvm.alloca %17 x i16 {alignment = 2 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%45, %7, %7, %7, %106, %107) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %108 = llvm.load %107 {alignment = 16 : i64} : !llvm.ptr -> vector<8xi16>
    %109 = llvm.bitcast %108 : vector<8xi16> to vector<8xf16>
    %110 = llvm.insertelement %98, %13[%0 : i32] : vector<2xi32>
    %111 = llvm.insertelement %104, %110[%1 : i32] : vector<2xi32>
    %112 = llvm.alloca %17 x i16 {alignment = 2 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%45, %7, %7, %7, %111, %112) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %113 = llvm.load %112 {alignment = 16 : i64} : !llvm.ptr -> vector<8xi16>
    %114 = llvm.bitcast %113 : vector<8xi16> to vector<8xf16>
    %115 = llvm.add %75, %76 : i64
    %116 = llvm.trunc %115 : i64 to i32
    %117 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %118 = llvm.insertelement %116, %117[%1 : i32] : vector<2xi32>
    %119 = llvm.alloca %17 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(%65, %15, %5, %15, %118, %119) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %120 = llvm.load %119 {alignment = 32 : i64} : !llvm.ptr -> vector<8xi32>
    %121 = llvm.bitcast %120 : vector<8xi32> to vector<16xf16>
    %122 = llvm.add %115, %11 : i64
    %123 = llvm.trunc %122 : i64 to i32
    %124 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %125 = llvm.insertelement %123, %124[%1 : i32] : vector<2xi32>
    %126 = llvm.alloca %17 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(%65, %15, %5, %15, %125, %126) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %127 = llvm.load %126 {alignment = 32 : i64} : !llvm.ptr -> vector<8xi32>
    %128 = llvm.bitcast %127 : vector<8xi32> to vector<16xf16>
    %129 = llvm.bitcast %96 : vector<8xf16> to vector<8xi16>
    %130 = llvm.bitcast %121 : vector<16xf16> to vector<8xi32>
    %131 = llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%129, %130, %77) {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %132 = llvm.bitcast %103 : vector<8xf16> to vector<8xi16>
    %133 = llvm.bitcast %128 : vector<16xf16> to vector<8xi32>
    %134 = llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%132, %133, %131) {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %135 = llvm.bitcast %109 : vector<8xf16> to vector<8xi16>
    %136 = llvm.bitcast %121 : vector<16xf16> to vector<8xi32>
    %137 = llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%135, %136, %78) {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %138 = llvm.bitcast %114 : vector<8xf16> to vector<8xi16>
    %139 = llvm.bitcast %128 : vector<16xf16> to vector<8xi32>
    %140 = llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%138, %139, %137) {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    cf.br ^bb1(%80, %134, %140 : i64, vector<8xf32>, vector<8xf32>)
  ^bb3:  // pred: ^bb1
    %141 = llvm.insertelement %19, %4[%3 : i64] : vector<4xi64>
    %142 = llvm.bitcast %141 : vector<4xi64> to vector<8xi32>
    %143 = llvm.insertelement %7, %142[%6 : i64] : vector<8xi32>
    %144 = llvm.insertelement %7, %143[%8 : i64] : vector<8xi32>
    %145 = llvm.insertelement %7, %144[%9 : i64] : vector<8xi32>
    %146 = llvm.bitcast %145 : vector<8xi32> to vector<4xi64>
    %147 = llvm.extractelement %146[%3 : i64] : vector<4xi64>
    %148 = llvm.trunc %72 : i64 to i32
    %149 = llvm.inttoptr %147 : i64 to !llvm.ptr<1>
    %150 = llvm.bitcast %77 : vector<8xf32> to vector<8xi32>
    %151 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %152 = llvm.insertelement %148, %151[%1 : i32] : vector<2xi32>
    %153 = llvm.alloca %17 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %150, %153 {alignment = 32 : i64} : vector<8xi32>, !llvm.ptr
    llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%149, %18, %7, %18, %152, %153) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) -> ()
    %154 = llvm.trunc %73 : i64 to i32
    %155 = llvm.bitcast %78 : vector<8xf32> to vector<8xi32>
    %156 = llvm.insertelement %63, %13[%0 : i32] : vector<2xi32>
    %157 = llvm.insertelement %154, %156[%1 : i32] : vector<2xi32>
    %158 = llvm.alloca %17 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %155, %158 {alignment = 32 : i64} : vector<8xi32>, !llvm.ptr
    llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%149, %18, %7, %18, %157, %158) {no_unwind, will_return} : (!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) -> ()
    return
  }
}
