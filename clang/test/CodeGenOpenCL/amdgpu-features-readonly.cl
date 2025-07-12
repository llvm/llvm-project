// RUN: %clang_cc1 -triple amdgcn -target-feature +vmem-to-lds-load-insts -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=READONLY %s

// READONLY: warning: feature flag '+vmem-to-lds-load-insts' is ignored since the feature is read only [-Winvalid-command-line-argument]

kernel void test() {}
