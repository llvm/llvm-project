// RUN: fir-opt --function-attr="unique-internal-linkage-names=true" %s | FileCheck %s --check-prefix=ENABLED
// RUN: fir-opt --function-attr="unique-internal-linkage-names=false" %s | FileCheck %s --check-prefix=DISABLED

// ENABLED: func.func @_QPtest() {
// ENABLED-NEXT:   return
// ENABLED-NEXT: }

// DISABLED: func.func @_QPtest() {
// DISABLED-NEXT:   return
// DISABLED-NEXT: }
func.func @_QPtest() {
  return
}

// ENABLED: func.func @_QFtestPfoo() attributes {fir.host_symbol = @_QPtest, llvm.sample_profile_suffix_elision_policy = "selected"} {
// ENABLED-NEXT:   return
// ENABLED-NEXT: }

// DISABLED: func.func @_QFtestPfoo() attributes {fir.host_symbol = @_QPtest} {
// DISABLED-NEXT:   return
// DISABLED-NEXT: }

func.func @_QFtestPfoo() attributes {fir.host_symbol = @_QPtest} {
  return
}

