// RUN: cc -c %S/print_complex.c
// RUN: tco %s | llc | as -o %t 
// RUN: cc %t print_complex.o 
// RUN: ./a.out | FileCheck %s

// CHECK: <0.935893, 2.252526>

func @foo(%a : !fir.complex<4>, %b : !fir.complex<4>, %c : !fir.complex<4>, %d : !fir.complex<4>, %e : !fir.complex<4>) -> !fir.complex<4> {
  %1 = fir.addc %a, %b : !fir.complex<4>
  %2 = fir.mulc %1, %c : !fir.complex<4>
  %3 = fir.subc %2, %d : !fir.complex<4>
  %4 = fir.divc %3, %e : !fir.complex<4>
  return %4 : !fir.complex<4>
}

func @real_part(%a : !fir.complex<4>) -> f32 {
  %0 = constant 0 : i32
  %1 = fir.extract_value %a, %0 : (!fir.complex<4>, i32) -> f32
  return %1 : f32
}

func @conj(%a : !fir.complex<4>) -> !fir.complex<4> {
  %0 = constant 1 : i32
  %1 = fir.extract_value %a, %0 : (!fir.complex<4>, i32) -> f32
  %2 = fir.negf %1 : f32
  %3 = fir.insert_value %a, %2, %0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  return %3 : !fir.complex<4>
}

func @print_complex(f32, f32)

func @main() -> i32 {
  %0 = fir.alloca !fir.complex<4>
  %1 = fir.alloca !fir.complex<4>
  %2 = fir.alloca !fir.complex<4>
  %3 = fir.alloca !fir.complex<4>
  %4 = fir.alloca !fir.complex<4>

  %5 = fir.undefined !fir.complex<4>
  %c0 = constant 0 : i32
  %c1 = constant 1 : i32

  %f0 = constant 4.0 : f32
  %f1 = constant 52.5 : f32
  %6 = fir.insert_value %5, %f0, %c0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  %7 = fir.insert_value %6, %f1, %c1 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  fir.store %7 to %0 : !fir.ref<!fir.complex<4>>

  %8 = fir.load %0 : !fir.ref<!fir.complex<4>>
  %9 = fir.call @conj(%8) : (!fir.complex<4>) -> !fir.complex<4>
  fir.store %9 to %1 : !fir.ref<!fir.complex<4>>

  %a0 = constant 95.65 : f32
  %a1 = constant 234.1 : f32
  %a2 = fir.insert_value %5, %a0, %c0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  %a3 = fir.insert_value %a2, %a1, %c1 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  fir.store %a3 to %2 : !fir.ref<!fir.complex<4>>

  %b0 = constant 33.0 : f32
  %b1 = constant 87.69 : f32
  %b2 = fir.insert_value %5, %b0, %c0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  %b3 = fir.insert_value %b2, %b1, %c1 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  fir.store %b3 to %3 : !fir.ref<!fir.complex<4>>

  %d0 = constant 791.0 : f32
  %d1 = constant 3.5923 : f32
  %d2 = fir.insert_value %5, %d0, %c0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  %d3 = fir.insert_value %d2, %d1, %c1 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  fir.store %d3 to %4 : !fir.ref<!fir.complex<4>>

  %l0 = fir.load %0 : !fir.ref<!fir.complex<4>>
  %l1 = fir.load %1 : !fir.ref<!fir.complex<4>>
  %l2 = fir.load %2 : !fir.ref<!fir.complex<4>>
  %l3 = fir.load %3 : !fir.ref<!fir.complex<4>>
  %l4 = fir.load %4 : !fir.ref<!fir.complex<4>>

  %10 = fir.call @foo(%l0, %l1, %l2, %l3, %l4) : (!fir.complex<4>, !fir.complex<4>, !fir.complex<4>, !fir.complex<4>, !fir.complex<4>) -> !fir.complex<4>
  %11 = fir.extract_value %10, %c0 : (!fir.complex<4>, i32) -> f32
  %12 = fir.extract_value %10, %c1 : (!fir.complex<4>, i32) -> f32
  fir.call @print_complex(%11, %12) : (f32, f32) -> ()
  return %c0 : i32
}
