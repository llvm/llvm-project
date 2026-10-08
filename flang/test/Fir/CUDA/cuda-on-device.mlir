// RUN: fir-opt --cuf-convert %s | FileCheck %s --check-prefix=FOLD
// RUN: fir-opt --cuf-convert-late %s | FileCheck %s --check-prefix=FOLD
// RUN: fir-opt --cuf-convert="defer-acc-routine-data-transfers=true" %s | FileCheck %s --check-prefix=DEFER

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  func.func @host() -> i1 {
    %0 = cuf.on_device : i1
    return %0 : i1
  }

  func.func @host_device() -> i1 attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
    %0 = cuf.on_device : i1
    return %0 : i1
  }

  func.func @device_proc() -> i1 attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
    %0 = cuf.on_device : i1
    return %0 : i1
  }

  func.func @launch() -> i1 {
    %c1 = arith.constant 1 : index
    %0 = arith.constant false
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
      %1 = cuf.on_device : i1
      gpu.terminator
    }
    return %0 : i1
  }

  gpu.module @cuda_device_mod {
    gpu.func @on_device_kernel() -> i1 {
      %0 = cuf.on_device : i1
      gpu.return %0 : i1
    }
  }

  func.func @acc_routine() -> i1 attributes {acc.routine_info = #acc.routine_info<[@routine]>} {
    %0 = cuf.on_device : i1
    return %0 : i1
  }

  func.func @device_specialized() -> i1 attributes {acc.specialized_routine = #acc.specialized_routine<@routine, <seq>, "device_specialized">} {
    %0 = acc.compute_region -> i1 {
      %1 = cuf.on_device : i1
      acc.yield %1 : i1
    } <{origin = "acc.routine"}>
    return %0 : i1
  }
}

// FOLD-LABEL: func.func @host()
// FOLD: %[[FALSE:.*]] = arith.constant false
// FOLD: return %[[FALSE]] : i1
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @host()
// DEFER: arith.constant false
// DEFER-NOT: cuf.on_device

// FOLD-LABEL: func.func @host_device()
// FOLD: arith.constant false
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @host_device()
// DEFER: arith.constant false
// DEFER-NOT: cuf.on_device

// FOLD-LABEL: func.func @device_proc()
// FOLD: arith.constant true
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @device_proc()
// DEFER: arith.constant true
// DEFER-NOT: cuf.on_device

// FOLD-LABEL: func.func @launch()
// FOLD: gpu.launch
// FOLD: arith.constant true
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @launch()
// DEFER: gpu.launch
// DEFER: arith.constant true
// DEFER-NOT: cuf.on_device

// FOLD-LABEL: gpu.func @on_device_kernel()
// FOLD: arith.constant true
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: gpu.func @on_device_kernel()
// DEFER: arith.constant true
// DEFER-NOT: cuf.on_device

// FOLD-LABEL: func.func @acc_routine()
// FOLD: arith.constant false
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @acc_routine()
// DEFER: cuf.on_device : i1
// DEFER-NOT: arith.constant

// FOLD-LABEL: func.func @device_specialized()
// FOLD: arith.constant true
// FOLD-NOT: cuf.on_device

// DEFER-LABEL: func.func @device_specialized()
// DEFER: arith.constant true
// DEFER-NOT: cuf.on_device
