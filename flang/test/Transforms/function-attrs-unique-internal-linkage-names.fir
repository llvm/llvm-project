// RUN: fir-opt --function-attr="unique-internal-linkage-names=true" %s | FileCheck %s --check-prefix=ENABLED
// RUN: fir-opt --function-attr="unique-internal-linkage-names=false" %s | FileCheck %s --check-prefix=DISABLED

// Internal procedure: has fir.host_symbol, should get the attribute.
// ENABLED-LABEL: func.func @_QFhost_subPinner(
// ENABLED-SAME: llvm.sample_profile_suffix_elision_policy = "selected"

// DISABLED-LABEL: func.func @_QFhost_subPinner(
// DISABLED-NOT: sample_profile_suffix_elision_policy
func.func @_QFhost_subPinner(%arg0: !fir.ref<i32>) attributes {fir.host_symbol = @_QFhost_sub} {
  return
}

// Host procedure: no fir.host_symbol, should NOT get the attribute.
// ENABLED-LABEL: func.func @_QFhost_sub(
// ENABLED-NOT: sample_profile_suffix_elision_policy
func.func @_QFhost_sub(%arg0: !fir.ref<i32>) {
  return
}

// External procedure: should NOT get the attribute.
// ENABLED-LABEL: func.func @_QPexternal_sub(
// ENABLED-NOT: sample_profile_suffix_elision_policy
func.func @_QPexternal_sub(%arg0: !fir.ref<i32>) {
  return
}
