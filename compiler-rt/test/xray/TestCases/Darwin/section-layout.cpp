// Verify XRay Mach-O section layout: xray_instr_map in __DATA segment.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: otool -l %t | FileCheck %s --check-prefix SECTION
// RUN: otool -l %t | FileCheck %s --check-prefix FNIDX

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// SECTION:      sectname xray_instr_map
// SECTION-NEXT:  segname __DATA
// FNIDX:        sectname xray_fn_idx
// FNIDX-NEXT:    segname __DATA

[[clang::xray_always_instrument]] void section_check_fn() {}

int main() {
  section_check_fn();
  return 0;
}
