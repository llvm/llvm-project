// RUN: fir-opt -split-input-file --omp-function-filtering %s | FileCheck %s

// CHECK:     func.func @any
// CHECK:     func.func @nohost
// CHECK-NOT: func.func @host
// CHECK-NOT: func.func @none
// CHECK:     func.func @nohost_target
// CHECK:     func.func @host_target
// CHECK:     func.func @none_target
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

// CHECK:     func.func @any
// CHECK-NOT: func.func @nohost
// CHECK:     func.func @host
// CHECK:     func.func @none
// CHECK:     func.func @nohost_target
// CHECK:     func.func @host_target
// CHECK:     func.func @none_target
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
