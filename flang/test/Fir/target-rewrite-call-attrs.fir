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

// The attributes of the arguments passed unchanged follow them when other
// arguments are split (complex<f64> into two f64), moved to the end (character
// length), or a hidden result argument is added in front.

func.func private @split(!fir.box<!fir.array<?xf32>>, complex<f64>, !fir.box<!fir.array<?xf32>>, !fir.boxchar<1>, !fir.box<!fir.array<?xf32>>)
func.func private @sret(!fir.box<!fir.array<?xf32>>) -> !fir.type<t{a:!fir.array<5xf64>}>

func.func @shifted(%b0: !fir.box<!fir.array<?xf32>>, %c: complex<f64>, %b1: !fir.box<!fir.array<?xf32>>, %ch: !fir.boxchar<1>, %b2: !fir.box<!fir.array<?xf32>>) {
  fir.call @split(%b0, %c, %b1, %ch, %b2) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, complex<f64>, !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<managed>}, !fir.boxchar<1>, !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<unified>}) -> ()
  return
}

func.func @hidden_result(%b0: !fir.box<!fir.array<?xf32>>) {
  %r = fir.call @sret(%b0) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}) -> !fir.type<t{a:!fir.array<5xf64>}>
  return
}

// CHECK-LABEL: func.func @shifted(
// CHECK: fir.call @split(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) proc_attrs<bind_c> : (!fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, f64, f64, !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<managed>}, !fir.ref<!fir.char<1,?>>, !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<unified>}, i64) -> ()

// CHECK-LABEL: func.func @hidden_result(
// CHECK: fir.call @sret(%{{.*}}, %{{.*}}) proc_attrs<bind_c> : (!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t{a:!fir.array<5xf64>}>}, !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}) -> ()
