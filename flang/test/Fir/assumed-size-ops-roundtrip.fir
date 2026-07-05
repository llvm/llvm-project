// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @roundtrip() {
  // CHECK: %[[E:.*]] = fir.assumed_size_extent : index
  %e = fir.assumed_size_extent : index

  // CHECK: %[[T:.*]] = fir.is_assumed_size_extent %[[E]] : (index) -> i1
  %t = fir.is_assumed_size_extent %e : (index) -> i1

  return
}


