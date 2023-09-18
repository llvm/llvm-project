// RUN: fir-opt -split-input-file --omp-function-filtering %s | FileCheck %s

// CHECK: func.func @any
// CHECK: return
// CHECK: func.func @nohost
// CHECK: return
// CHECK: func.func private @host
// CHECK-NOT: return
// CHECK: func.func private @none
// CHECK-NOT: return
// CHECK: func.func @nohost_target
// CHECK: return
// CHECK: func.func @host_target
// CHECK: return
// CHECK: func.func @none_target
// CHECK: return
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
  func.func @none_target() -> () {
    omp.target {}
    func.return
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
