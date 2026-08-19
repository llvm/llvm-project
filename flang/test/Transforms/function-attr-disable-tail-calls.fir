// RUN: fir-opt --function-attr="disable-tail-calls=true" %s | FileCheck %s --check-prefix=ENABLED
// RUN: fir-opt --function-attr="disable-tail-calls=false" %s | FileCheck %s --check-prefix=DISABLED

func.func @_QPf() {
  return
}

// ENABLED: func.func @_QPf() attributes {llvm.disable_tail_calls = true} {
// ENABLED-NEXT:   return
// ENABLED-NEXT: }

// DISABLED: func.func @_QPf() {
// DISABLED-NOT: llvm.disable_tail_calls
// DISABLED-NEXT:   return
// DISABLED-NEXT: }
