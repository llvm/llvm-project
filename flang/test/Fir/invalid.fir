// RUN: fir-opt -split-input-file -verify-diagnostics %s

// expected-error@+2{{expected integer value}}
// expected-error@+1{{kind value expected}}
func private @it1() -> !fir.int<A>

// -----

// expected-error@+1{{custom op 'fir.string_lit' must have character type}}
%0 = fir.string_lit "Hello, World!"(13) : !fir.int<32>

// -----

// expected-error@+1{{custom op 'fir.string_lit' found an invalid constant}}
%0 = fir.string_lit 20(13) : !fir.int<32>

// -----

// expected-error@+1{{'fir.string_lit' op values in list must be integers}}
%2 = fir.string_lit [158, 2.0](2) : !fir.char<2>

// -----

func @bad_rebox_1(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op operand #0 must be The type of a Fortran descriptor, but got '!fir.ref<!fir.array<?x?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_2(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op result #0 must be The type of a Fortran descriptor, but got '!fir.ref<!fir.array<?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_3(%arg0: !fir.box<!fir.array<*:f32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op box operand must not have unknown rank or type}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<*:f32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_4(%arg0: !fir.box<!fir.array<?xf32>>) {
  // expected-error@+1{{op result type must not have unknown rank or type}}
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<*:f32>>
  return
}

// -----

func @bad_rebox_5(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1{{op slice operand rank must match box operand rank}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_6(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shift %c1, %c1 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_7(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{shape operand must absent or be a fir.shift when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_8(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %undef = fir.undefined index
  %0 = fir.slice %c1, %undef, %undef, %c1, %c10, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // expected-error@+1{{result type rank and rank after applying slice operand must match}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func @bad_rebox_9(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shift %c10, %c10 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when the shape is a fir.shift}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func @bad_rebox_10(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  // expected-error@+1{{result type and shape operand ranks must match}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_11(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c42 = constant 42 : index
  %0 = fir.shape %c42 : (index) -> !fir.shape<1>
  // expected-error@+1{{op input and output element types must match for intrinsic types}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  return
}
