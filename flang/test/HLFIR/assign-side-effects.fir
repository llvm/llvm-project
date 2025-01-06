// Test side effects of hlfir.assign op.
// RUN: fir-opt %s --test-side-effects --verify-diagnostics

func.func @test1(%x: !fir.ref<i32>, %i: i32) {
  // expected-remark @below {{found an instance of 'write' on a op operand, on resource '<Default>'}}
  hlfir.assign %i to %x : i32, !fir.ref<i32>
  return
}

func.func @test2(%x: !fir.ref<i32>, %y: !fir.ref<i32>) {
  // expected-remark @below {{found an instance of 'write' on a op operand, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on a op operand, on resource '<Default>'}}
  hlfir.assign %y to %x : !fir.ref<i32>, !fir.ref<i32>
  return
}

func.func @test3(%x: !fir.ref<!fir.type<t>>, %y: !fir.ref<!fir.type<t>>) {
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on resource '<Default>'}}
  hlfir.assign %y to %x : !fir.ref<!fir.type<t>>, !fir.ref<!fir.type<t>>
  return
}

func.func @test4(%x: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %y: !fir.box<!fir.array<?xi32>>) {
  // expected-remark @below {{found an instance of 'read' on a op operand, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'free' on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'allocate' on resource '<Default>'}}
  hlfir.assign %y to %x realloc : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
