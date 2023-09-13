// RUN: fir-opt %s --test-side-effects --verify-diagnostics

func.func @concat(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1, 20>>) {
// expected-remark@+1 {{operation has no memory effects}}
  %c30 = arith.constant 30 : index
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @all_no_effects(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
// expected-remark@+1 {{operation has no memory effects}}
  %all = hlfir.all %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @all_effects(%arg0: !fir.ref<!fir.array<2x10x!fir.logical<4>>>, %arg1: i32) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %all = hlfir.all %arg0 dim %arg1 : (!fir.ref<!fir.array<2x10x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @any_no_effects(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
// expected-remark@+1 {{operation has no memory effects}}
  %all = hlfir.any %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @any_effects(%arg0: !fir.ref<!fir.array<2x10x!fir.logical<4>>>, %arg1: i32) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %all = hlfir.any %arg0 dim %arg1 : (!fir.ref<!fir.array<2x10x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @count_no_effects(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
// expected-remark@+1 {{operation has no memory effects}}
  %all = hlfir.count %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> i32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @count_effects(%arg0: !fir.ref<!fir.array<2x10x!fir.logical<4>>>, %arg1: i32) {
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %all = hlfir.count %arg0 dim %arg1 : (!fir.ref<!fir.array<2x10x!fir.logical<4>>>, i32) -> i32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @product_no_effects(%arg0: !hlfir.expr<?xf32>) {
// expected-remark@+1 {{operation has no memory effects}}
  %product = hlfir.product %arg0 : (!hlfir.expr<?xf32>) -> f32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @product_effects(%arg0: !fir.ref<!fir.array<2x2xf32>>, %arg1: i32) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %product = hlfir.product %arg0 dim %arg1 : (!fir.ref<!fir.array<2x2xf32>>, i32) -> !hlfir.expr<2xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @set_length_read(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: index) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.set_length %arg0 len %arg1 : (!fir.ref<!fir.char<1,10>>, index) -> !hlfir.expr<!fir.char<1,?>>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @sum_no_effects(%arg0: !hlfir.expr<?xf32>) {
// expected-remark@+1 {{operation has no memory effects}}
  %sum = hlfir.sum %arg0 : (!hlfir.expr<?xf32>) -> f32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @sum_effects(%arg0: !fir.ref<!fir.array<2x2xf32>>, %arg1: i32) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %sum = hlfir.sum %arg0 dim %arg1 : (!fir.ref<!fir.array<2x2xf32>>, i32) -> !hlfir.expr<2xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @maxval_no_effects(%arg0: !hlfir.expr<?xf32>) {
// expected-remark@+1 {{operation has no memory effects}}
  %maxval = hlfir.maxval %arg0 : (!hlfir.expr<?xf32>) -> f32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @maxval_effects(%arg0: !fir.ref<!fir.array<2x2xf32>>, %arg1: i32) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %maxval = hlfir.maxval %arg0 dim %arg1 : (!fir.ref<!fir.array<2x2xf32>>, i32) -> !hlfir.expr<2xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @dot_product_no_effects(%arg0: !hlfir.expr<?xf32>, %arg1: !hlfir.expr<?xf32>) {
// expected-remark@+1 {{operation has no memory effects}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xf32>, !hlfir.expr<?xf32>) -> f32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @dot_product_effects(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>) {
// there are read effects on both arguments - the diagnostic verification just doesn't register duplicate identical diagnostics
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) -> f32
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @matmul_no_reads(%arg0: !hlfir.expr<?x?xf32>, %arg1: !hlfir.expr<?x?xf32>) {
// expected-remark@+1 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xf32>, !hlfir.expr<?x?xf32>) -> !hlfir.expr<?x?xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @matmul_reads(%arg0: !fir.ref<!fir.array<10x5xf32>>, %arg1: !fir.ref<!fir.array<5x10xf32>>) {
// expected-remark@+3 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// there are read effects on both arguments - the diagnostic verification just doesn't register duplicate identical diagnostics
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.matmul %arg0 %arg1 : (!fir.ref<!fir.array<10x5xf32>>, !fir.ref<!fir.array<5x10xf32>>) -> !hlfir.expr<10x10xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @transpose_no_reads(%arg0: !hlfir.expr<?x?xf32>) {
// expected-remark@+1 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
  %0 = hlfir.transpose %arg0 : (!hlfir.expr<?x?xf32>) -> !hlfir.expr<?x?xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @transpose_read(%arg0: !fir.ref<!fir.array<10x5xf32>>) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.transpose %arg0 : (!fir.ref<!fir.array<10x5xf32>>) -> !hlfir.expr<5x10xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @matmul_transpose_no_reads(%arg0: !hlfir.expr<?x?xf32>, %arg1: !hlfir.expr<?x?xf32>) {
// expected-remark@+1 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?xf32>, !hlfir.expr<?x?xf32>) -> !hlfir.expr<?x?xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @matmul_transpose_reads(%arg0: !fir.ref<!fir.array<5x10xf32>>, %arg1: !fir.ref<!fir.array<5x10xf32>>) {
// expected-remark@+3 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// there are read effects on both arguments - the diagnostic verification just doesn't register duplicate identical diagnostics
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!fir.ref<!fir.array<5x10xf32>>, !fir.ref<!fir.array<5x10xf32>>) -> !hlfir.expr<10x10xf32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @associate(%arg0: i32) {
// expected-remark@+1 {{found an instance of 'allocate' on resource '<Default>'}}
  %0:3 = hlfir.associate %arg0 {uniq_name = "x"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// expected-remark@+1 {{found an instance of 'free' on resource '<Default>'}}
  hlfir.end_associate %0#1, %0#2 : !fir.ref<i32>, i1
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @as_expr_read(%arg0: !fir.ref<!fir.array<2xi32>>) {
// expected-remark@+2 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.as_expr %arg0 : (!fir.ref<!fir.array<2xi32>>) -> !hlfir.expr<?xi32>
// expected-remark@+1 {{found an instance of 'free' on resource '<Default>'}}
  hlfir.destroy %0 : !hlfir.expr<?xi32>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @char_extremum(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>) {
// expected-remark@+3 {{found an instance of 'allocate' on a value, on resource '<Default>'}}
// there are read effects on both arguments - the diagnostic verification just doesn't register duplicate identical diagnostics
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.ref<!fir.char<1, 10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @copy_in(%box: !fir.box<!fir.array<?xf64>>, %is_present: i1) {
// expected-remark@+2 {{found an instance of 'allocate' on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0:2 = hlfir.copy_in %box : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
// expected-remark@+1 {{operation has no memory effects}}
  return
}

func.func @copy_out(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.box<!fir.array<?xf64>>, %was_copied: i1) {
// expected-remark@+2 {{found an instance of 'free' on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  hlfir.copy_out %temp, %was_copied : (!fir.box<!fir.array<?xf64>>, i1) -> ()
// expected-remark@+3 {{found an instance of 'free' on resource '<Default>'}}
// expected-remark@+2 {{found an instance of 'read' on a value, on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'write' on a value, on resource '<Default>'}}
  hlfir.copy_out %temp, %was_copied to %box : (!fir.box<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
// expected-remark@+1 {{operation has no memory effects}}
  return
}
