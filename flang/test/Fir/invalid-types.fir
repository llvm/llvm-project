// Test the FIR types parser diagnostics
// RUN: fir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{expected non-function type}}
func private @box3() -> !fir.boxproc<>

// -----

// expected-error@+2 {{expected attribute value}}
// expected-error@+1 {{expected affine map}}
func private @box1() -> !fir.box<!fir.array<?xf32>, >

// -----

// expected-error@+1 {{expected non-function type}}
func private @box1() -> !fir.box<>

// -----

// expected-error@+1 {{expected integer value}}
func private @box2() -> !fir.boxchar<>

// -----

// expected-error@+1 {{expected integer value}}
func private @it6() -> !fir.char<>

// -----

// expected-error@+1 {{expected integer value}}
func private @it6() -> !fir.char<2, >

// -----

// expected-error@+1 {{expected integer value}}
func private @it3() -> !fir.complex<>

// -----

// expected-error@+1 {{expected non-function type}}
func private @mem3() -> !fir.heap<>

// -----

// expected-error@+1 {{expected integer value}}
func private @it1() -> !fir.int<A>

// -----

// expected-error@+1 {{expected integer value}}
func private @it1() -> !fir.logical<b>

// -----

// expected-error@+1 {{expected non-function type}}
func private @mem3() -> !fir.ptr<>

// -----

// expected-error@+1 {{expected integer value}}
func private @mem3() -> !fir.real<>

// -----

// expected-error@+1 {{expected valid keyword}}
func private @mem3() -> !fir.type<>

// -----

// expected-error@+2 {{expected valid keyword}}
// expected-error@+1 {{expected LEN parameter list}}
func private @dvd4() -> !fir.type<derived4()>

// -----

// expected-error@+2 {{expected ':'}}
// expected-error@+1 {{expected LEN parameter list}}
func private @dvd4() -> !fir.type<derived4(p)>

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{expected LEN parameter list}}
func private @dvd4() -> !fir.type<derived4(p:)>

// -----

// expected-error@+2 {{expected valid keyword}}
// expected-error@+1 {{expected field type list}}
func private @dvd4() -> !fir.type<derived4(p:i8){}>

// -----

// expected-error@+2 {{expected ':'}}
// expected-error@+1 {{expected field type list}}
func private @dvd4() -> !fir.type<derived4(p:i8){f1}>

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{expected field type list}}
func private @dvd4() -> !fir.type<derived4(p:i8){f1:f2}>

// -----

// expected-error@+2 {{expected valid keyword}}
// expected-error@+1 {{expected field type list}}
func private @dvd4() -> !fir.type<derived4(p:i8){f1:i32,}>

// -----

// expected-error@+2 {{expected valid keyword}}
// expected-error@+1 {{expected field type list}}
func private @dvd4() -> !fir.type<derived4(p:i8){,}>

// -----

// expected-error@+1 {{expected non-function type}}
func private @mem3() -> !fir.ref<>

// -----

// expected-error@+1 {{expected ':'}}
func private @arr1() -> !fir.array<*>

// -----

// expected-error@+1 {{expected non-function type}}
func private @arr1() -> !fir.array<*:>

// -----

// expected-error@+1 {{expected integer value}}
func private @oth1() -> !fir.shape<>

// -----

// expected-error@+1 {{expected integer value}}
func private @oth1() -> !fir.shapeshift<>

// -----

// expected-error@+1 {{expected integer value}}
func private @oth1() -> !fir.shift<>

// -----

// expected-error@+1 {{expected integer value}}
func private @oth1() -> !fir.slice<>

// -----

// expected-error@+1 {{expected non-function type}}
func private @oth3() -> !fir.tdesc<xx>

// -----

// expected-error@+1 {{expected integer value}}
func private @oth3() -> !fir.vector<>

// -----

// expected-error@+1 {{expected ':'}}
func private @oth3() -> !fir.vector<10>

// -----

// expected-error@+1 {{expected non-function type}}
func private @oth3() -> !fir.vector<10:>
