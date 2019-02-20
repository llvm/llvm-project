// RUN: %clang_cc1 -E -fxray-instrument %s -o - | FileCheck --check-prefix=CHECK-XRAY %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-XRAY %s

#if __has_feature(xray_instrument)
int XRayInstrumentEnabled();
#else
int XRayInstrumentDisabled();
#endif

// CHECK-XRAY: XRayInstrumentEnabled
// CHECK-NO-XRAY: XRayInstrumentDisabled
