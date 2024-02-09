// RUN: fir-opt -split-input-file --omp-function-filtering %s | FileCheck %s

// CHECK: func.func @any
// CHECK: return
// CHECK: func.func @nohost
// CHECK: return
// CHECK-NOT: func.func {{.*}}}} @host
// CHECK-NOT: func.func {{.*}}}} @none
// CHECK: func.func @nohost_target
// CHECK: return
// CHECK: func.func @host_target
// CHECK: return
// CHECK: func.func @none_target
// CHECK: return
// CHECK: func.func @host_target_call
// CHECK-NOT: call @none_target
// CHECK: %[[UNDEF:.*]] = fir.undefined i32
// CHECK: return %[[UNDEF]] : i32
module attributes {omp.is_target_device = true} {
  func.func @any() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (any), capture_clause = (to)>
      } {
    func.return
  }
  func.func @nohost() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    func.return
  }
  func.func @host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    func.return
  }
  func.func @none() -> () {
    func.return
  }
  func.func @nohost_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    omp.target {}
    func.return
  }
  func.func @host_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    omp.target {}
    func.return
  }
  func.func @none_target() -> i32 {
    omp.target {}
    %0 = arith.constant 25 : i32
    func.return %0 : i32
  }
  func.func @host_target_call() -> i32
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    omp.target {}
    %0 = call @none_target() : () -> i32
    func.return %0 : i32
  }
}

// -----

// CHECK: func.func @any
// CHECK: return
// CHECK: func.func @nohost
// CHECK: return
// CHECK: func.func @host
// CHECK: return
// CHECK: func.func @none
// CHECK: return
// CHECK: func.func @nohost_target
// CHECK: return
// CHECK: func.func @host_target
// CHECK: return
// CHECK: func.func @none_target
// CHECK: return
module attributes {omp.is_target_device = false} {
  func.func @any() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (any), capture_clause = (to)>
      } {
    func.return
  }
  func.func @nohost() -> ()
      attributes {
          omp.declare_target =
            #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    func.return
  }
  func.func @host() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    func.return
  }
  func.func @none() -> () {
    func.return
  }
  func.func @nohost_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (nohost), capture_clause = (to)>
      } {
    omp.target {}
    func.return
  }
  func.func @host_target() -> ()
      attributes {
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    omp.target {}
    func.return
  }
  func.func @none_target() -> () {
    omp.target {}
    func.return
  }
}
