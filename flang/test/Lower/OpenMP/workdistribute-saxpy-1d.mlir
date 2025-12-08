module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:mooxiu/llvm-project.git a8a0ffba739d247e24faaf612ac8f2d8faf1de3c)", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.target_triples = [], omp.version = #omp.version<version = 60>} {
  func.func @_QPtarget_teams_workdistribute() {
    %c9 = arith.constant 9 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFtarget_teams_workdistributeEa"}
    %2 = fir.declare %1 {uniq_name = "_QFtarget_teams_workdistributeEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
    %4 = fir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_int_kind"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_logical_kind) : !fir.ref<i32>
    %6 = fir.declare %5 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_logical_kind"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.address_of(@_QMiso_fortran_env_implECbfloat16) : !fir.ref<i32>
    %8 = fir.declare %7 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECbfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.address_of(@_QMiso_fortran_envECcharacter_kinds) : !fir.ref<!fir.array<3xi32>>
    %10 = fir.shape %c3 : (index) -> !fir.shape<1>
    %11 = fir.declare %9(%10) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcharacter_kinds"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<3xi32>>
    %12 = fir.address_of(@_QMiso_fortran_envECcharacter_storage_size) : !fir.ref<i32>
    %13 = fir.declare %12 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcharacter_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %14 = fir.address_of(@_QMiso_fortran_envECcurrent_team) : !fir.ref<i32>
    %15 = fir.declare %14 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcurrent_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %16 = fir.address_of(@_QMiso_fortran_envECerror_unit) : !fir.ref<i32>
    %17 = fir.declare %16 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECerror_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %18 = fir.address_of(@_QMiso_fortran_envECfile_storage_size) : !fir.ref<i32>
    %19 = fir.declare %18 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECfile_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %20 = fir.address_of(@_QMiso_fortran_envECinitial_team) : !fir.ref<i32>
    %21 = fir.declare %20 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECinitial_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %22 = fir.address_of(@_QMiso_fortran_envECinput_unit) : !fir.ref<i32>
    %23 = fir.declare %22 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECinput_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %24 = fir.address_of(@_QMiso_fortran_env_implECint128) : !fir.ref<i32>
    %25 = fir.declare %24 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %26 = fir.address_of(@_QMiso_fortran_env_implECint16) : !fir.ref<i32>
    %27 = fir.declare %26 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %28 = fir.address_of(@_QMiso_fortran_env_implECint32) : !fir.ref<i32>
    %29 = fir.declare %28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %30 = fir.address_of(@_QMiso_fortran_env_implECint64) : !fir.ref<i32>
    %31 = fir.declare %30 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %32 = fir.address_of(@_QMiso_fortran_env_implECint8) : !fir.ref<i32>
    %33 = fir.declare %32 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %34 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_integer_kinds) : !fir.ref<!fir.array<5xi32>>
    %35 = fir.shape %c5 : (index) -> !fir.shape<1>
    %36 = fir.declare %34(%35) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_integer_kinds"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
    %37 = fir.address_of(@_QMiso_fortran_envECiostat_end) : !fir.ref<i32>
    %38 = fir.declare %37 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_end"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %39 = fir.address_of(@_QMiso_fortran_envECiostat_eor) : !fir.ref<i32>
    %40 = fir.declare %39 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_eor"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %41 = fir.address_of(@_QMiso_fortran_envECiostat_inquire_internal_unit) : !fir.ref<i32>
    %42 = fir.declare %41 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_inquire_internal_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %43 = fir.address_of(@_QMiso_fortran_env_implEClogical16) : !fir.ref<i32>
    %44 = fir.declare %43 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %45 = fir.address_of(@_QMiso_fortran_env_implEClogical32) : !fir.ref<i32>
    %46 = fir.declare %45 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %47 = fir.address_of(@_QMiso_fortran_env_implEClogical64) : !fir.ref<i32>
    %48 = fir.declare %47 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %49 = fir.address_of(@_QMiso_fortran_env_implEClogical8) : !fir.ref<i32>
    %50 = fir.declare %49 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %51 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_logical_kinds) : !fir.ref<!fir.array<4xi32>>
    %52 = fir.shape %c4 : (index) -> !fir.shape<1>
    %53 = fir.declare %51(%52) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_logical_kinds"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xi32>>
    %54 = fir.address_of(@_QMiso_fortran_envECnumeric_storage_size) : !fir.ref<i32>
    %55 = fir.declare %54 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECnumeric_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %56 = fir.address_of(@_QMiso_fortran_envECoutput_unit) : !fir.ref<i32>
    %57 = fir.declare %56 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECoutput_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %58 = fir.address_of(@_QMiso_fortran_envECparent_team) : !fir.ref<i32>
    %59 = fir.declare %58 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECparent_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %60 = fir.address_of(@_QMiso_fortran_env_implECreal128) : !fir.ref<i32>
    %61 = fir.declare %60 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %62 = fir.address_of(@_QMiso_fortran_env_implECreal16) : !fir.ref<i32>
    %63 = fir.declare %62 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %64 = fir.address_of(@_QMiso_fortran_env_implECreal32) : !fir.ref<i32>
    %65 = fir.declare %64 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %66 = fir.address_of(@_QMiso_fortran_env_implECreal64) : !fir.ref<i32>
    %67 = fir.declare %66 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %68 = fir.address_of(@_QMiso_fortran_env_implECreal64x2) : !fir.ref<i32>
    %69 = fir.declare %68 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %70 = fir.address_of(@_QMiso_fortran_env_implECreal80) : !fir.ref<i32>
    %71 = fir.declare %70 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %72 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_real_kinds) : !fir.ref<!fir.array<5xi32>>
    %73 = fir.declare %72(%35) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_real_kinds"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
    %74 = fir.address_of(@_QMiso_fortran_env_implECsafebfloat16) : !fir.ref<i32>
    %75 = fir.declare %74 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafebfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %76 = fir.address_of(@_QMiso_fortran_env_implECsafeint128) : !fir.ref<i32>
    %77 = fir.declare %76 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %78 = fir.address_of(@_QMiso_fortran_env_implECsafeint16) : !fir.ref<i32>
    %79 = fir.declare %78 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %80 = fir.address_of(@_QMiso_fortran_env_implECsafeint32) : !fir.ref<i32>
    %81 = fir.declare %80 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %82 = fir.address_of(@_QMiso_fortran_env_implECsafeint64) : !fir.ref<i32>
    %83 = fir.declare %82 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %84 = fir.address_of(@_QMiso_fortran_env_implECsafeint8) : !fir.ref<i32>
    %85 = fir.declare %84 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %86 = fir.address_of(@_QMiso_fortran_env_implECsafereal128) : !fir.ref<i32>
    %87 = fir.declare %86 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %88 = fir.address_of(@_QMiso_fortran_env_implECsafereal16) : !fir.ref<i32>
    %89 = fir.declare %88 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %90 = fir.address_of(@_QMiso_fortran_env_implECsafereal32) : !fir.ref<i32>
    %91 = fir.declare %90 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %92 = fir.address_of(@_QMiso_fortran_env_implECsafereal64) : !fir.ref<i32>
    %93 = fir.declare %92 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %94 = fir.address_of(@_QMiso_fortran_env_implECsafereal64x2) : !fir.ref<i32>
    %95 = fir.declare %94 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %96 = fir.address_of(@_QMiso_fortran_env_implECsafereal80) : !fir.ref<i32>
    %97 = fir.declare %96 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %98 = fir.address_of(@_QMiso_fortran_env_implECsafeuint128) : !fir.ref<i32>
    %99 = fir.declare %98 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %100 = fir.address_of(@_QMiso_fortran_env_implECsafeuint16) : !fir.ref<i32>
    %101 = fir.declare %100 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %102 = fir.address_of(@_QMiso_fortran_env_implECsafeuint32) : !fir.ref<i32>
    %103 = fir.declare %102 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %104 = fir.address_of(@_QMiso_fortran_env_implECsafeuint64) : !fir.ref<i32>
    %105 = fir.declare %104 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %106 = fir.address_of(@_QMiso_fortran_env_implECsafeuint8) : !fir.ref<i32>
    %107 = fir.declare %106 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %108 = fir.address_of(@_QMiso_fortran_env_implECselectedbfloat16) : !fir.ref<i32>
    %109 = fir.declare %108 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedbfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %110 = fir.address_of(@_QMiso_fortran_env_implECselectedint128) : !fir.ref<i32>
    %111 = fir.declare %110 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %112 = fir.address_of(@_QMiso_fortran_env_implECselectedint16) : !fir.ref<i32>
    %113 = fir.declare %112 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %114 = fir.address_of(@_QMiso_fortran_env_implECselectedint32) : !fir.ref<i32>
    %115 = fir.declare %114 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %116 = fir.address_of(@_QMiso_fortran_env_implECselectedint64) : !fir.ref<i32>
    %117 = fir.declare %116 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %118 = fir.address_of(@_QMiso_fortran_env_implECselectedint8) : !fir.ref<i32>
    %119 = fir.declare %118 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %120 = fir.address_of(@_QMiso_fortran_env_implECselectedreal128) : !fir.ref<i32>
    %121 = fir.declare %120 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %122 = fir.address_of(@_QMiso_fortran_env_implECselectedreal16) : !fir.ref<i32>
    %123 = fir.declare %122 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %124 = fir.address_of(@_QMiso_fortran_env_implECselectedreal32) : !fir.ref<i32>
    %125 = fir.declare %124 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %126 = fir.address_of(@_QMiso_fortran_env_implECselectedreal64) : !fir.ref<i32>
    %127 = fir.declare %126 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %128 = fir.address_of(@_QMiso_fortran_env_implECselectedreal64x2) : !fir.ref<i32>
    %129 = fir.declare %128 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %130 = fir.address_of(@_QMiso_fortran_env_implECselectedreal80) : !fir.ref<i32>
    %131 = fir.declare %130 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %132 = fir.address_of(@_QMiso_fortran_env_implECselecteduint128) : !fir.ref<i32>
    %133 = fir.declare %132 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %134 = fir.address_of(@_QMiso_fortran_env_implECselecteduint16) : !fir.ref<i32>
    %135 = fir.declare %134 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %136 = fir.address_of(@_QMiso_fortran_env_implECselecteduint32) : !fir.ref<i32>
    %137 = fir.declare %136 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %138 = fir.address_of(@_QMiso_fortran_env_implECselecteduint64) : !fir.ref<i32>
    %139 = fir.declare %138 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %140 = fir.address_of(@_QMiso_fortran_env_implECselecteduint8) : !fir.ref<i32>
    %141 = fir.declare %140 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %142 = fir.address_of(@_QMiso_fortran_envECstat_failed_image) : !fir.ref<i32>
    %143 = fir.declare %142 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_failed_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %144 = fir.address_of(@_QMiso_fortran_envECstat_locked) : !fir.ref<i32>
    %145 = fir.declare %144 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_locked"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %146 = fir.address_of(@_QMiso_fortran_envECstat_locked_other_image) : !fir.ref<i32>
    %147 = fir.declare %146 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_locked_other_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %148 = fir.address_of(@_QMiso_fortran_envECstat_stopped_image) : !fir.ref<i32>
    %149 = fir.declare %148 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_stopped_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %150 = fir.address_of(@_QMiso_fortran_envECstat_unlocked) : !fir.ref<i32>
    %151 = fir.declare %150 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_unlocked"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %152 = fir.address_of(@_QMiso_fortran_envECstat_unlocked_failed_image) : !fir.ref<i32>
    %153 = fir.declare %152 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_unlocked_failed_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %154 = fir.address_of(@_QMiso_fortran_env_implECuint128) : !fir.ref<i32>
    %155 = fir.declare %154 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %156 = fir.address_of(@_QMiso_fortran_env_implECuint16) : !fir.ref<i32>
    %157 = fir.declare %156 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %158 = fir.address_of(@_QMiso_fortran_env_implECuint32) : !fir.ref<i32>
    %159 = fir.declare %158 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %160 = fir.address_of(@_QMiso_fortran_env_implECuint64) : !fir.ref<i32>
    %161 = fir.declare %160 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %162 = fir.address_of(@_QMiso_fortran_env_implECuint8) : !fir.ref<i32>
    %163 = fir.declare %162 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %164 = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtarget_teams_workdistributeEx"}
    %165 = fir.shape %c10 : (index) -> !fir.shape<1>
    %166 = fir.declare %164(%165) {uniq_name = "_QFtarget_teams_workdistributeEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %167 = fir.alloca !fir.array<10xf32> {bindc_name = "y", uniq_name = "_QFtarget_teams_workdistributeEy"}
    %168 = fir.declare %167(%165) {uniq_name = "_QFtarget_teams_workdistributeEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %169 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%c9 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
    %170 = omp.map.info var_ptr(%168 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%169) -> !fir.ref<!fir.array<10xf32>> {name = "y"}
    %171 = omp.map.info var_ptr(%2 : !fir.ref<f32>, f32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<f32> {name = "a"}
    %172 = omp.map.info var_ptr(%166 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%169) -> !fir.ref<!fir.array<10xf32>> {name = "x"}
    %173 = omp.map.info var_ptr(%168 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(storage) capture(ByRef) bounds(%169) -> !fir.ref<!fir.array<10xf32>> {name = "y"}
    %174 = omp.map.info var_ptr(%2 : !fir.ref<f32>, f32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<f32> {name = "a"}
    %175 = omp.map.info var_ptr(%166 : !fir.ref<!fir.array<10xf32>>, !fir.array<10xf32>) map_clauses(storage) capture(ByRef) bounds(%169) -> !fir.ref<!fir.array<10xf32>> {name = "x"}
    omp.target_data map_entries(%170, %172 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
      %176 = fir.alloca f32
      %177 = omp.map.info var_ptr(%176 : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "__flang_workdistribute_from"}
      %178 = omp.map.info var_ptr(%176 : !fir.ref<f32>, f32) map_clauses(to) capture(ByRef) -> !fir.ref<f32> {name = "__flang_workdistribute_to"}
      %179 = llvm.mlir.constant(0 : i32) : i32
      %c1_0 = arith.constant 1 : index
      %c10_1 = arith.constant 10 : index
      %180 = fir.shape %c10_1 : (index) -> !fir.shape<1>
      %181 = fir.declare %168(%180) {uniq_name = "_QFtarget_teams_workdistributeEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
      %182 = fir.declare %2 {uniq_name = "_QFtarget_teams_workdistributeEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
      %183 = fir.declare %166(%180) {uniq_name = "_QFtarget_teams_workdistributeEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
      %184 = fir.load %182 : !fir.ref<f32>
      fir.store %184 to %176 : !fir.ref<f32>
      omp.target host_eval(%c1_0 -> %arg0, %c10_1 -> %arg1, %c1_0 -> %arg2 : index, index, index) map_entries(%173 -> %arg3, %174 -> %arg4, %175 -> %arg5, %178 -> %arg6 : !fir.ref<!fir.array<10xf32>>, !fir.ref<f32>, !fir.ref<!fir.array<10xf32>>, !fir.ref<f32>) {
        %185 = fir.load %arg6 : !fir.ref<f32>
        %c1_2 = arith.constant 1 : index
        %c10_3 = arith.constant 10 : index
        %186 = fir.shape %c10_3 : (index) -> !fir.shape<1>
        %187 = fir.declare %arg3(%186) {uniq_name = "_QFtarget_teams_workdistributeEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
        %188 = fir.declare %arg5(%186) {uniq_name = "_QFtarget_teams_workdistributeEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
        omp.teams {
          omp.parallel {
            omp.distribute {
              omp.wsloop {
                omp.loop_nest (%arg7) : index = (%arg0) to (%arg1) inclusive step (%arg2) {
                  %189 = fir.array_coor %188(%186) %arg7 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
                  %190 = fir.load %189 : !fir.ref<f32>
                  %191 = arith.mulf %185, %190 fastmath<contract> : f32
                  %192 = fir.array_coor %187(%186) %arg7 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
                  %193 = fir.load %192 : !fir.ref<f32>
                  %194 = arith.addf %191, %193 fastmath<contract> : f32
                  %195 = fir.array_coor %187(%186) %arg7 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
                  fir.store %194 to %195 : !fir.ref<f32>
                  omp.yield
                }
              } {omp.composite}
            } {omp.composite}
            omp.terminator
          } {omp.composite}
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
  func.func @_QPteams_workdistribute() {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFteams_workdistributeEa"}
    %2 = fir.declare %1 {uniq_name = "_QFteams_workdistributeEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
    %4 = fir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_int_kind"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_logical_kind) : !fir.ref<i32>
    %6 = fir.declare %5 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_atomic_logical_kind"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.address_of(@_QMiso_fortran_env_implECbfloat16) : !fir.ref<i32>
    %8 = fir.declare %7 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECbfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.address_of(@_QMiso_fortran_envECcharacter_kinds) : !fir.ref<!fir.array<3xi32>>
    %10 = fir.shape %c3 : (index) -> !fir.shape<1>
    %11 = fir.declare %9(%10) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcharacter_kinds"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<3xi32>>
    %12 = fir.address_of(@_QMiso_fortran_envECcharacter_storage_size) : !fir.ref<i32>
    %13 = fir.declare %12 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcharacter_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %14 = fir.address_of(@_QMiso_fortran_envECcurrent_team) : !fir.ref<i32>
    %15 = fir.declare %14 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcurrent_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %16 = fir.address_of(@_QMiso_fortran_envECerror_unit) : !fir.ref<i32>
    %17 = fir.declare %16 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECerror_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %18 = fir.address_of(@_QMiso_fortran_envECfile_storage_size) : !fir.ref<i32>
    %19 = fir.declare %18 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECfile_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %20 = fir.address_of(@_QMiso_fortran_envECinitial_team) : !fir.ref<i32>
    %21 = fir.declare %20 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECinitial_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %22 = fir.address_of(@_QMiso_fortran_envECinput_unit) : !fir.ref<i32>
    %23 = fir.declare %22 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECinput_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %24 = fir.address_of(@_QMiso_fortran_env_implECint128) : !fir.ref<i32>
    %25 = fir.declare %24 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %26 = fir.address_of(@_QMiso_fortran_env_implECint16) : !fir.ref<i32>
    %27 = fir.declare %26 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %28 = fir.address_of(@_QMiso_fortran_env_implECint32) : !fir.ref<i32>
    %29 = fir.declare %28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %30 = fir.address_of(@_QMiso_fortran_env_implECint64) : !fir.ref<i32>
    %31 = fir.declare %30 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %32 = fir.address_of(@_QMiso_fortran_env_implECint8) : !fir.ref<i32>
    %33 = fir.declare %32 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %34 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_integer_kinds) : !fir.ref<!fir.array<5xi32>>
    %35 = fir.shape %c5 : (index) -> !fir.shape<1>
    %36 = fir.declare %34(%35) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_integer_kinds"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
    %37 = fir.address_of(@_QMiso_fortran_envECiostat_end) : !fir.ref<i32>
    %38 = fir.declare %37 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_end"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %39 = fir.address_of(@_QMiso_fortran_envECiostat_eor) : !fir.ref<i32>
    %40 = fir.declare %39 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_eor"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %41 = fir.address_of(@_QMiso_fortran_envECiostat_inquire_internal_unit) : !fir.ref<i32>
    %42 = fir.declare %41 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECiostat_inquire_internal_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %43 = fir.address_of(@_QMiso_fortran_env_implEClogical16) : !fir.ref<i32>
    %44 = fir.declare %43 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %45 = fir.address_of(@_QMiso_fortran_env_implEClogical32) : !fir.ref<i32>
    %46 = fir.declare %45 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %47 = fir.address_of(@_QMiso_fortran_env_implEClogical64) : !fir.ref<i32>
    %48 = fir.declare %47 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %49 = fir.address_of(@_QMiso_fortran_env_implEClogical8) : !fir.ref<i32>
    %50 = fir.declare %49 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEClogical8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %51 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_logical_kinds) : !fir.ref<!fir.array<4xi32>>
    %52 = fir.shape %c4 : (index) -> !fir.shape<1>
    %53 = fir.declare %51(%52) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_logical_kinds"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xi32>>
    %54 = fir.address_of(@_QMiso_fortran_envECnumeric_storage_size) : !fir.ref<i32>
    %55 = fir.declare %54 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECnumeric_storage_size"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %56 = fir.address_of(@_QMiso_fortran_envECoutput_unit) : !fir.ref<i32>
    %57 = fir.declare %56 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECoutput_unit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %58 = fir.address_of(@_QMiso_fortran_envECparent_team) : !fir.ref<i32>
    %59 = fir.declare %58 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECparent_team"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %60 = fir.address_of(@_QMiso_fortran_env_implECreal128) : !fir.ref<i32>
    %61 = fir.declare %60 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %62 = fir.address_of(@_QMiso_fortran_env_implECreal16) : !fir.ref<i32>
    %63 = fir.declare %62 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %64 = fir.address_of(@_QMiso_fortran_env_implECreal32) : !fir.ref<i32>
    %65 = fir.declare %64 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %66 = fir.address_of(@_QMiso_fortran_env_implECreal64) : !fir.ref<i32>
    %67 = fir.declare %66 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %68 = fir.address_of(@_QMiso_fortran_env_implECreal64x2) : !fir.ref<i32>
    %69 = fir.declare %68 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %70 = fir.address_of(@_QMiso_fortran_env_implECreal80) : !fir.ref<i32>
    %71 = fir.declare %70 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECreal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %72 = fir.address_of(@_QMiso_fortran_env_implEC__builtin_real_kinds) : !fir.ref<!fir.array<5xi32>>
    %73 = fir.declare %72(%35) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implEC__builtin_real_kinds"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xi32>>
    %74 = fir.address_of(@_QMiso_fortran_env_implECsafebfloat16) : !fir.ref<i32>
    %75 = fir.declare %74 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafebfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %76 = fir.address_of(@_QMiso_fortran_env_implECsafeint128) : !fir.ref<i32>
    %77 = fir.declare %76 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %78 = fir.address_of(@_QMiso_fortran_env_implECsafeint16) : !fir.ref<i32>
    %79 = fir.declare %78 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %80 = fir.address_of(@_QMiso_fortran_env_implECsafeint32) : !fir.ref<i32>
    %81 = fir.declare %80 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %82 = fir.address_of(@_QMiso_fortran_env_implECsafeint64) : !fir.ref<i32>
    %83 = fir.declare %82 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %84 = fir.address_of(@_QMiso_fortran_env_implECsafeint8) : !fir.ref<i32>
    %85 = fir.declare %84 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %86 = fir.address_of(@_QMiso_fortran_env_implECsafereal128) : !fir.ref<i32>
    %87 = fir.declare %86 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %88 = fir.address_of(@_QMiso_fortran_env_implECsafereal16) : !fir.ref<i32>
    %89 = fir.declare %88 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %90 = fir.address_of(@_QMiso_fortran_env_implECsafereal32) : !fir.ref<i32>
    %91 = fir.declare %90 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %92 = fir.address_of(@_QMiso_fortran_env_implECsafereal64) : !fir.ref<i32>
    %93 = fir.declare %92 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %94 = fir.address_of(@_QMiso_fortran_env_implECsafereal64x2) : !fir.ref<i32>
    %95 = fir.declare %94 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %96 = fir.address_of(@_QMiso_fortran_env_implECsafereal80) : !fir.ref<i32>
    %97 = fir.declare %96 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafereal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %98 = fir.address_of(@_QMiso_fortran_env_implECsafeuint128) : !fir.ref<i32>
    %99 = fir.declare %98 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %100 = fir.address_of(@_QMiso_fortran_env_implECsafeuint16) : !fir.ref<i32>
    %101 = fir.declare %100 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %102 = fir.address_of(@_QMiso_fortran_env_implECsafeuint32) : !fir.ref<i32>
    %103 = fir.declare %102 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %104 = fir.address_of(@_QMiso_fortran_env_implECsafeuint64) : !fir.ref<i32>
    %105 = fir.declare %104 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %106 = fir.address_of(@_QMiso_fortran_env_implECsafeuint8) : !fir.ref<i32>
    %107 = fir.declare %106 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECsafeuint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %108 = fir.address_of(@_QMiso_fortran_env_implECselectedbfloat16) : !fir.ref<i32>
    %109 = fir.declare %108 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedbfloat16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %110 = fir.address_of(@_QMiso_fortran_env_implECselectedint128) : !fir.ref<i32>
    %111 = fir.declare %110 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %112 = fir.address_of(@_QMiso_fortran_env_implECselectedint16) : !fir.ref<i32>
    %113 = fir.declare %112 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %114 = fir.address_of(@_QMiso_fortran_env_implECselectedint32) : !fir.ref<i32>
    %115 = fir.declare %114 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %116 = fir.address_of(@_QMiso_fortran_env_implECselectedint64) : !fir.ref<i32>
    %117 = fir.declare %116 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %118 = fir.address_of(@_QMiso_fortran_env_implECselectedint8) : !fir.ref<i32>
    %119 = fir.declare %118 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %120 = fir.address_of(@_QMiso_fortran_env_implECselectedreal128) : !fir.ref<i32>
    %121 = fir.declare %120 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %122 = fir.address_of(@_QMiso_fortran_env_implECselectedreal16) : !fir.ref<i32>
    %123 = fir.declare %122 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %124 = fir.address_of(@_QMiso_fortran_env_implECselectedreal32) : !fir.ref<i32>
    %125 = fir.declare %124 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %126 = fir.address_of(@_QMiso_fortran_env_implECselectedreal64) : !fir.ref<i32>
    %127 = fir.declare %126 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %128 = fir.address_of(@_QMiso_fortran_env_implECselectedreal64x2) : !fir.ref<i32>
    %129 = fir.declare %128 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal64x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %130 = fir.address_of(@_QMiso_fortran_env_implECselectedreal80) : !fir.ref<i32>
    %131 = fir.declare %130 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselectedreal80"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %132 = fir.address_of(@_QMiso_fortran_env_implECselecteduint128) : !fir.ref<i32>
    %133 = fir.declare %132 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %134 = fir.address_of(@_QMiso_fortran_env_implECselecteduint16) : !fir.ref<i32>
    %135 = fir.declare %134 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %136 = fir.address_of(@_QMiso_fortran_env_implECselecteduint32) : !fir.ref<i32>
    %137 = fir.declare %136 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %138 = fir.address_of(@_QMiso_fortran_env_implECselecteduint64) : !fir.ref<i32>
    %139 = fir.declare %138 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %140 = fir.address_of(@_QMiso_fortran_env_implECselecteduint8) : !fir.ref<i32>
    %141 = fir.declare %140 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECselecteduint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %142 = fir.address_of(@_QMiso_fortran_envECstat_failed_image) : !fir.ref<i32>
    %143 = fir.declare %142 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_failed_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %144 = fir.address_of(@_QMiso_fortran_envECstat_locked) : !fir.ref<i32>
    %145 = fir.declare %144 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_locked"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %146 = fir.address_of(@_QMiso_fortran_envECstat_locked_other_image) : !fir.ref<i32>
    %147 = fir.declare %146 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_locked_other_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %148 = fir.address_of(@_QMiso_fortran_envECstat_stopped_image) : !fir.ref<i32>
    %149 = fir.declare %148 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_stopped_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %150 = fir.address_of(@_QMiso_fortran_envECstat_unlocked) : !fir.ref<i32>
    %151 = fir.declare %150 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_unlocked"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %152 = fir.address_of(@_QMiso_fortran_envECstat_unlocked_failed_image) : !fir.ref<i32>
    %153 = fir.declare %152 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECstat_unlocked_failed_image"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %154 = fir.address_of(@_QMiso_fortran_env_implECuint128) : !fir.ref<i32>
    %155 = fir.declare %154 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint128"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %156 = fir.address_of(@_QMiso_fortran_env_implECuint16) : !fir.ref<i32>
    %157 = fir.declare %156 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint16"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %158 = fir.address_of(@_QMiso_fortran_env_implECuint32) : !fir.ref<i32>
    %159 = fir.declare %158 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint32"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %160 = fir.address_of(@_QMiso_fortran_env_implECuint64) : !fir.ref<i32>
    %161 = fir.declare %160 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint64"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %162 = fir.address_of(@_QMiso_fortran_env_implECuint8) : !fir.ref<i32>
    %163 = fir.declare %162 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_env_implECuint8"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %164 = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFteams_workdistributeEx"}
    %165 = fir.shape %c10 : (index) -> !fir.shape<1>
    %166 = fir.declare %164(%165) {uniq_name = "_QFteams_workdistributeEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %167 = fir.alloca !fir.array<10xf32> {bindc_name = "y", uniq_name = "_QFteams_workdistributeEy"}
    %168 = fir.declare %167(%165) {uniq_name = "_QFteams_workdistributeEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %169 = fir.load %2 : !fir.ref<f32>
    omp.teams {
      omp.parallel {
        omp.distribute {
          omp.wsloop {
            omp.loop_nest (%arg0) : index = (%c1) to (%c10) inclusive step (%c1) {
              %170 = fir.array_coor %166(%165) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
              %171 = fir.load %170 : !fir.ref<f32>
              %172 = arith.mulf %169, %171 fastmath<contract> : f32
              %173 = fir.array_coor %168(%165) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
              %174 = fir.load %173 : !fir.ref<f32>
              %175 = arith.addf %172, %174 fastmath<contract> : f32
              %176 = fir.array_coor %168(%165) %arg0 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
              fir.store %175 to %176 : !fir.ref<f32>
              omp.yield
            }
          } {omp.composite}
        } {omp.composite}
        omp.terminator
      } {omp.composite}
      omp.terminator
    }
    return
  }
  fir.global @_QM__fortran_builtinsEC__builtin_atomic_int_kind constant : i32
  fir.global @_QM__fortran_builtinsEC__builtin_atomic_logical_kind constant : i32
  fir.global @_QMiso_fortran_env_implECbfloat16 constant : i32
  fir.global @_QMiso_fortran_envECcharacter_kinds constant : !fir.array<3xi32>
  fir.global @_QMiso_fortran_envECcharacter_storage_size constant : i32
  fir.global @_QMiso_fortran_envECcurrent_team constant : i32
  fir.global @_QMiso_fortran_envECerror_unit constant : i32
  fir.global @_QMiso_fortran_envECfile_storage_size constant : i32
  fir.global @_QMiso_fortran_envECinitial_team constant : i32
  fir.global @_QMiso_fortran_envECinput_unit constant : i32
  fir.global @_QMiso_fortran_env_implECint128 constant : i32
  fir.global @_QMiso_fortran_env_implECint16 constant : i32
  fir.global @_QMiso_fortran_env_implECint32 constant : i32
  fir.global @_QMiso_fortran_env_implECint64 constant : i32
  fir.global @_QMiso_fortran_env_implECint8 constant : i32
  fir.global @_QMiso_fortran_env_implEC__builtin_integer_kinds constant : !fir.array<5xi32>
  fir.global @_QMiso_fortran_envECiostat_end constant : i32
  fir.global @_QMiso_fortran_envECiostat_eor constant : i32
  fir.global @_QMiso_fortran_envECiostat_inquire_internal_unit constant : i32
  fir.global @_QMiso_fortran_env_implEClogical16 constant : i32
  fir.global @_QMiso_fortran_env_implEClogical32 constant : i32
  fir.global @_QMiso_fortran_env_implEClogical64 constant : i32
  fir.global @_QMiso_fortran_env_implEClogical8 constant : i32
  fir.global @_QMiso_fortran_env_implEC__builtin_logical_kinds constant : !fir.array<4xi32>
  fir.global @_QMiso_fortran_envECnumeric_storage_size constant : i32
  fir.global @_QMiso_fortran_envECoutput_unit constant : i32
  fir.global @_QMiso_fortran_envECparent_team constant : i32
  fir.global @_QMiso_fortran_env_implECreal128 constant : i32
  fir.global @_QMiso_fortran_env_implECreal16 constant : i32
  fir.global @_QMiso_fortran_env_implECreal32 constant : i32
  fir.global @_QMiso_fortran_env_implECreal64 constant : i32
  fir.global @_QMiso_fortran_env_implECreal64x2 constant : i32
  fir.global @_QMiso_fortran_env_implECreal80 constant : i32
  fir.global @_QMiso_fortran_env_implEC__builtin_real_kinds constant : !fir.array<5xi32>
  fir.global @_QMiso_fortran_env_implECsafebfloat16 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeint128 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeint16 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeint32 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeint64 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeint8 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal128 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal16 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal32 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal64 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal64x2 constant : i32
  fir.global @_QMiso_fortran_env_implECsafereal80 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeuint128 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeuint16 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeuint32 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeuint64 constant : i32
  fir.global @_QMiso_fortran_env_implECsafeuint8 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedbfloat16 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedint128 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedint16 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedint32 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedint64 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedint8 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal128 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal16 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal32 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal64 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal64x2 constant : i32
  fir.global @_QMiso_fortran_env_implECselectedreal80 constant : i32
  fir.global @_QMiso_fortran_env_implECselecteduint128 constant : i32
  fir.global @_QMiso_fortran_env_implECselecteduint16 constant : i32
  fir.global @_QMiso_fortran_env_implECselecteduint32 constant : i32
  fir.global @_QMiso_fortran_env_implECselecteduint64 constant : i32
  fir.global @_QMiso_fortran_env_implECselecteduint8 constant : i32
  fir.global @_QMiso_fortran_envECstat_failed_image constant : i32
  fir.global @_QMiso_fortran_envECstat_locked constant : i32
  fir.global @_QMiso_fortran_envECstat_locked_other_image constant : i32
  fir.global @_QMiso_fortran_envECstat_stopped_image constant : i32
  fir.global @_QMiso_fortran_envECstat_unlocked constant : i32
  fir.global @_QMiso_fortran_envECstat_unlocked_failed_image constant : i32
  fir.global @_QMiso_fortran_env_implECuint128 constant : i32
  fir.global @_QMiso_fortran_env_implECuint16 constant : i32
  fir.global @_QMiso_fortran_env_implECuint32 constant : i32
  fir.global @_QMiso_fortran_env_implECuint64 constant : i32
  fir.global @_QMiso_fortran_env_implECuint8 constant : i32
}
