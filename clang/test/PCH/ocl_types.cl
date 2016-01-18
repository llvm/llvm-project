// Test this without pch.
// RUN: %clang_cc1 -include %S/ocl_types.h -fsyntax-only %s

// Test with pch.
// RUN: %clang_cc1 -x cl -emit-pch -o %t %S/ocl_types.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s -ast-print

void foo1(img1d_t img);

void foo2(img1darr_t img);

void foo3(img1dbuff_t img);

void foo4(img2d_t img);

void foo5(img2darr_t img);

void foo6(img3d_t img);

void foo7(smp_t smp) {
  smp_t loc_smp;
}

void foo8(evt_t evt) {
  evt_t loc_evt;
}
