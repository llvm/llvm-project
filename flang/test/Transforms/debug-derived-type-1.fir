// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Only enabled on x86_64
// REQUIRES: x86-registered-target

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.version = #omp.version<version = 11>} {
  fir.global @_QMm_employeeEemployee : !fir.type<_QMm_employeeTt_employee{t_person:!fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}>,hired_date:!fir.type<_QMm_employeeTt_date{year:i32,month:i32,day:i32}>,monthly_salary:f32}> {
    %0 = fir.zero_bits !fir.type<_QMm_employeeTt_employee{t_person:!fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}>,hired_date:!fir.type<_QMm_employeeTt_date{year:i32,month:i32,day:i32}>,monthly_salary:f32}>
    fir.has_value %0 : !fir.type<_QMm_employeeTt_employee{t_person:!fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}>,hired_date:!fir.type<_QMm_employeeTt_date{year:i32,month:i32,day:i32}>,monthly_salary:f32}>
  } loc(#loc5)
  fir.global @_QMt1Evar : !fir.type<_QMt1Tt_t1{age:i32,points:!fir.array<3x!fir.complex<4>>,cond:!fir.logical<1>,name:!fir.char<1,20>,ratio:f64}> {
    %0 = fir.zero_bits !fir.type<_QMt1Tt_t1{age:i32,points:!fir.array<3x!fir.complex<4>>,cond:!fir.logical<1>,name:!fir.char<1,20>,ratio:f64}>
    fir.has_value %0 : !fir.type<_QMt1Tt_t1{age:i32,points:!fir.array<3x!fir.complex<4>>,cond:!fir.logical<1>,name:!fir.char<1,20>,ratio:f64}>
  } loc(#loc6)
  fir.type_info @_QMt1Tt_t1 noinit nodestroy nofinal : !fir.type<_QMt1Tt_t1{age:i32,points:!fir.array<3x!fir.complex<4>>,cond:!fir.logical<1>,name:!fir.char<1,20>,ratio:f64}> loc(#loc7)
  fir.type_info @_QMm_employeeTt_address noinit nodestroy nofinal : !fir.type<_QMm_employeeTt_address{house_number:i32}> loc(#loc1)
  fir.type_info @_QMm_employeeTt_person noinit nodestroy nofinal extends !fir.type<_QMm_employeeTt_address{house_number:i32}> : !fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}> loc(#loc2)
  fir.type_info @_QMm_employeeTt_date noinit nodestroy nofinal : !fir.type<_QMm_employeeTt_date{year:i32,month:i32,day:i32}> loc(#loc3)
  fir.type_info @_QMm_employeeTt_employee noinit nodestroy nofinal extends !fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}> : !fir.type<_QMm_employeeTt_employee{t_person:!fir.type<_QMm_employeeTt_person{t_address:!fir.type<_QMm_employeeTt_address{house_number:i32}>,name:!fir.char<1,20>}>,hired_date:!fir.type<_QMm_employeeTt_date{year:i32,month:i32,day:i32}>,monthly_salary:f32}> loc(#loc4)
  fir.type_info @_QFTt_pair noinit nodestroy nofinal : !fir.type<_QFTt_pair{i:i64,x:f64}> loc(#loc8)
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %1 = fir.alloca !fir.type<_QFTt_pair{i:i64,x:f64}> {bindc_name = "pair", uniq_name = "_QFEpair"}
    %2 = fircg.ext_declare %1 {uniq_name = "_QFEpair"} : (!fir.ref<!fir.type<_QFTt_pair{i:i64,x:f64}>>) -> !fir.ref<!fir.type<_QFTt_pair{i:i64,x:f64}>> loc(#loc9)
    return
  } loc(#loc10)
}
#loc1 = loc("derived1.f90":24:1)
#loc2 = loc("derived1.f90":35:25)
#loc3 = loc("derived1.f90":17:1)
#loc4 = loc("derived1.f90":46:1)
#loc5 = loc("derived1.f90":50:3)
#loc6 = loc("derived1.f90":62:3)
#loc7 = loc("derived1.f90":70:3)
#loc8 = loc("derived1.f90":85:3)
#loc9 = loc("derived1.f90":77:3)
#loc10 = loc("derived1.f90":75:3)


// CHECK-DAG: #[[INT_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[INT8_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 64, encoding = DW_ATE_signed>
// CHECK-DAG: #[[REAL4_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[CMX8_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 64, encoding = DW_ATE_complex_float>
// CHECK-DAG: #[[CMX_ARR:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #[[CMX8_TY:.*]]{{.*}}>
// CHECK-DAG: #[[LOG_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "logical", sizeInBits = 8, encoding = DW_ATE_boolean>
// CHECK-DAG: #[[REAL8_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK-DAG: #[[STR_TY:.*]] = #llvm.di_string_type
// CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<{{.*}}name = "m_employee"{{.*}}>
// CHECK-DAG: #[[MOD1:.*]] = #llvm.di_module<{{.*}}name = "t1"{{.*}}>
// CHECK-DAG: #[[ELMA1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "house_number", baseType = #[[INT_TY]], sizeInBits = 32, alignInBits = 32>
// CHECK-DAG: #[[ADDR:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_address"{{.*}}line = 24, scope = #[[MOD]], sizeInBits = 32, elements = #[[ELMA1]]>
// CHECK-DAG: #[[ELMD1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "year", baseType = #[[INT_TY]], sizeInBits = 32, alignInBits = 32>
// CHECK-DAG: #[[ELMD2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "month", baseType = #[[INT_TY]], sizeInBits = 32, alignInBits = 32, offsetInBits = 32>
// CHECK-DAG: #[[ELMD3:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "day", baseType = #[[INT_TY]], sizeInBits = 32, alignInBits = 32, offsetInBits = 64>
// CHECK-DAG: #[[DATE:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_date", file = #di_file, line = 17, scope = #[[MOD]], sizeInBits = 96, elements = #[[ELMD1]], #[[ELMD2]], #[[ELMD3]]>
// CHECK-DAG: #[[ELMP1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "t_address", baseType = #[[ADDR]], sizeInBits = 32, alignInBits = 32>
// CHECK-DAG: #[[ELMP2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "name", baseType = #[[STR_TY]], sizeInBits = 160, alignInBits = 8, offsetInBits = 32>
// CHECK-DAG: #[[PERS:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_person"{{.*}}line = 35, scope = #[[MOD]], sizeInBits = 192, elements = #[[ELMP1]], #[[ELMP2]]>
// CHECK-DAG: #[[ELME1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "t_person", baseType = #[[PERS]], sizeInBits = 192, alignInBits = 32>
// CHECK-DAG: #[[ELME2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "hired_date", baseType = #[[DATE]], sizeInBits = 96, alignInBits = 32, offsetInBits = 192>
// CHECK-DAG: #[[ELME3:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "monthly_salary", baseType = #[[REAL4_TY]], sizeInBits = 32, alignInBits = 32, offsetInBits = 288>
// CHECK-DAG: #[[EMP:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_employee"{{.*}}line = 46, scope = #[[MOD]], sizeInBits = 320, elements = #[[ELME1]], #[[ELME2]], #[[ELME3]]>

// CHECK-DAG: #[[ELM1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "age", baseType = #[[INT_TY]], sizeInBits = 32, alignInBits = 32>
// CHECK-DAG: #[[ELM2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "points", baseType = #[[CMX_ARR]], sizeInBits = 192, alignInBits = 32, offsetInBits = 32>
// CHECK-DAG: #[[ELM3:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "cond", baseType = #[[LOG_TY]], sizeInBits = 8, alignInBits = 8, offsetInBits = 224>
// CHECK-DAG: #[[ELM4:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "name", baseType = #[[STR_TY]], sizeInBits = 160, alignInBits = 8, offsetInBits = 232>
// CHECK-DAG: #[[ELM5:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "ratio", baseType = #[[REAL8_TY]], sizeInBits = 64, alignInBits = 64, offsetInBits = 448>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_t1"{{.*}}, line = 70, scope = #[[MOD1]], sizeInBits = 512, elements = #[[ELM1]], #[[ELM2]], #[[ELM3]], #[[ELM4]], #[[ELM5]]>

// CHECK-DAG: #[[SP:.*]] = #llvm.di_subprogram
// CHECK-DAG: #[[ELML1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "i", baseType = #[[INT8_TY]], sizeInBits = 64, alignInBits = 64>
// CHECK-DAG: #[[ELML2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "x", baseType = #[[REAL8_TY]], sizeInBits = 64, alignInBits = 64, offsetInBits = 64>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "t_pair"{{.*}}line = 85, scope = #di_subprogram, sizeInBits = 128, elements = #[[ELML1]], #[[ELML2]]>
