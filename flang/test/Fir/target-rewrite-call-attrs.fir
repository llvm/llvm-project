// Test that rewriting a call keeps its procedure attributes and the attributes
// of the arguments passed unchanged.
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

func.func private @claunchc(!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, complex<f32>) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

func.func @direct(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: complex<f32>) {
  fir.call @claunchc(%arg0, %arg1) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, complex<f32>) -> ()
  return
}

func.func @indirect(%arg: () -> (), %arg0: !fir.box<!fir.array<?xf32>>, %arg1: complex<f32>) {
  %f = fir.convert %arg : (() -> ()) -> ((!fir.box<!fir.array<?xf32>>, complex<f32>) -> ())
  fir.call %f(%arg0, %arg1) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, complex<f32>) -> ()
  return
}

// CHECK-LABEL: func.func @direct(
// CHECK: fir.call @claunchc(%{{.*}}, %{{.*}}) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, !fir.vector<2:f32>) -> ()

// CHECK-LABEL: func.func @indirect(
// CHECK: fir.call %{{.*}}(%{{.*}}, %{{.*}}) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, !fir.vector<2:f32>) -> ()
