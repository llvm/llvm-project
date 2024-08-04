// XFAIL: *
// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// TODO we can lower these but we have no guarantee that the parent of
// omp.workshare supports multi-block regions, thus we fail for now.

func.func @wsfunc() {
  %a = fir.alloca i32
  omp.parallel {
    omp.workshare {
    ^bb1:
      %c1 = arith.constant 1 : i32
      cf.br ^bb3(%c1: i32)
    ^bb3(%arg1: i32):
      "test.test2"(%arg1) : (i32) -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func @wsfunc() {
  %a = fir.alloca i32
  omp.parallel {
    omp.workshare {
    ^bb1:
      %c1 = arith.constant 1 : i32
      cf.br ^bb3(%c1: i32)
    ^bb2:
      "test.test2"(%r) : (i32) -> ()
      omp.terminator
    ^bb3(%arg1: i32):
      %r = "test.test2"(%arg1) : (i32) -> i32
      cf.br ^bb2
    }
    omp.terminator
  }
  return
}
