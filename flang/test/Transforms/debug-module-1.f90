! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | \
! RUN: fir-opt --cg-rewrite --mlir-print-debuginfo | fir-opt --add-debug-info --mlir-print-debuginfo | FileCheck %s

! CHECK-DAG: #[[I4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
! CHECK-DAG: #[[R4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
! CHECK-DAG: #[[FILE:.*]] = #llvm.di_file<"debug-module-1.f90" {{.*}}>
! CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}, file = #[[FILE]], {{.*}}>
! CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<file = #[[FILE]], scope = #[[CU]], name = "helper", {{.*}}>
module helper
! CHECK-DAG: #[[LOC1:.*]] = loc("{{.*}}debug-module-1.f90":[[@LINE+2]]{{.*}})
! CHECK-DAG: #[[GLR:.*]] = #llvm.di_global_variable<scope = #[[MOD]], name = "glr", linkageName = "_QMhelperEglr", file = #[[FILE]], line = [[@LINE+1]], type = #[[R4]], isDefined = true>
  real glr
! CHECK-DAG: #[[LOC2:.*]] = loc("{{.*}}debug-module-1.f90":[[@LINE+2]]{{.*}})
! CHECK-DAG: #[[GLI:.*]] = #llvm.di_global_variable<scope = #[[MOD]], name = "gli", linkageName = "_QMhelperEgli", file = #[[FILE]], line = [[@LINE+1]], type = #[[I4]], isDefined = true>
  integer gli

  contains
! CHECK-DAG: #[[LOC3:.*]] = loc("{{.*}}debug-module-1.f90":[[@LINE+2]]{{.*}})
! CHECK-DAG: #[[TEST:.*]] = #llvm.di_subprogram<{{.*}}compileUnit = #[[CU]], scope = #[[MOD]], name = "test", linkageName = "_QMhelperPtest", file = #[[FILE]], line = [[@LINE+1]], scopeLine = [[@LINE+1]]{{.*}}>
    subroutine test()
    glr = 12.34
    gli = 67

    end subroutine
end module helper

program test
use helper
implicit none

  glr = 3.14
  gli = 2
  call test()

end program test

! CHECK-DAG: loc(fused<#[[GLR]]>[#[[LOC1]]])
! CHECK-DAG: loc(fused<#[[GLI]]>[#[[LOC2]]])
! CHECK-DAG: loc(fused<#[[TEST]]>[#[[LOC3]]])