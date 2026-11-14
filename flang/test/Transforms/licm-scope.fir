// RUN: fir-opt %s --flang-licm="only-inside=scf.execute_region" | FileCheck %s --check-prefix=SCOPED
// RUN: fir-opt %s --flang-licm="only-inside=scf.for" | FileCheck %s --check-prefix=SCOPED
// RUN: fir-opt %s --flang-licm > %t.default
// RUN: fir-opt %s --flang-licm="only-inside=func.func" > %t.function
// RUN: diff %t.default %t.function
// RUN: fir-opt %s > %t.input
// RUN: fir-opt %s --flang-licm="only-inside=scf.if" > %t.unmatched
// RUN: diff %t.input %t.unmatched
// RUN: fir-opt %s --flang-licm="only-inside=builtin.module" > %t.outside-function
// RUN: diff %t.input %t.outside-function

// A matching region can be nested under host control flow. Nested matching
// regions are supported; selecting a loop name does not select that loop itself.
// SCOPED-LABEL: func.func @nested
// SCOPED: scf.for
// SCOPED: arith.addi
// SCOPED: scf.execute_region
// SCOPED: scf.execute_region
// SCOPED: arith.muli
// SCOPED: scf.for
// SCOPED-NOT: arith.muli
// SCOPED: scf.yield
func.func @nested(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %outer = %c0 to %n step %c1 {
    %host = arith.addi %n, %n : index
    scf.execute_region {
      scf.execute_region {
        scf.for %inner = %c0 to %n step %c1 {
          %selected = arith.muli %n, %n : index
        }
        scf.yield
      }
      scf.yield
    }
  }
  return
}
