// Test the scope of an internal procedure, and of a target region it contains,
// when compiling for a target device. Such a module does not hold the host
// procedure they are contained in, so it must not be used as their scope
// there. The two inputs below only differ in omp.is_target_device.

// RUN: split-file %s %t
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/host.fir | FileCheck %s --check-prefix=HOST
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/device.fir | FileCheck %s --check-prefix=DEVICE

//--- host.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    return
  } loc(#loc1)
  func.func private @_QFPtar(%arg0: !fir.ref<i32>) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("n") -> !fir.ref<i32>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<i32>) {
      %1 = fircg.ext_declare %arg1 {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc3)
      omp.terminator
    } loc(#loc4)
    return
  } loc(#loc2)
}
#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":13:1)
#loc3 = loc("test.f90":15:1)
#loc4 = loc("test.f90":14:1)

// The host runs the host procedure, so it is the scope of both.
// HOST: #[[SP:.*]] = #llvm.di_subprogram<{{.*}}name = "test"{{.*}}>
// HOST-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SP]], name = "tar"{{.*}}>
// HOST-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SP]], name = "__omp_offloading_{{.*}}_QFPtar_l14"{{.*}}>

//--- device.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>, omp.is_gpu = true, omp.is_target_device = true} {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    return
  } loc(#loc1)
  func.func private @_QFPtar(%arg0: !fir.ref<i32>) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("n") -> !fir.ref<i32>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<i32>) {
      %1 = fircg.ext_declare %arg1 {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc3)
      omp.terminator
    } loc(#loc4)
    return
  } loc(#loc2)
}
#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":13:1)
#loc3 = loc("test.f90":15:1)
#loc4 = loc("test.f90":14:1)

// The device does not run the host procedure, so both are scoped at the file.
// DEVICE: #[[FILE:.*]] = #llvm.di_file<"device.fir"{{.*}}>
// DEVICE-DAG: #llvm.di_subprogram<{{.*}}scope = #[[FILE]], name = "tar"{{.*}}>
// DEVICE-DAG: #llvm.di_subprogram<{{.*}}scope = #[[FILE]], name = "__omp_offloading_{{.*}}_QFPtar_l14"{{.*}}>
