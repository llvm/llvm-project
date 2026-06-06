// RUN: %clang_cc1 %s -triple x86_64-scei-ps4 -DPS4 -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fsyntax-only -verify

#ifdef PS4

// PS4 does not support these.
void __vectorcall func_vc() {} // expected-error {{'__vectorcall' calling convention is not supported for this target}}
void __regcall func_rc() {} // expected-error {{'__regcall' calling convention is not supported for this target}}
void __attribute__((vectorcall)) funcA() {} // expected-error {{'vectorcall' calling convention is not supported for this target}}
void __attribute__((regcall)) funcB() {} // expected-error {{'regcall' calling convention is not supported for this target}}
void __attribute__((ms_abi)) funcH() {} // expected-error {{'ms_abi' calling convention is not supported for this target}}
void __attribute__((intel_ocl_bicc)) funcJ() {} // expected-error {{'intel_ocl_bicc' calling convention is not supported for this target}}
void __attribute__((swiftcall)) funcK() {} // expected-error {{'swiftcall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) funcKK() {} // expected-error {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((pascal)) funcG() {} // expected-error {{'pascal' calling convention is not supported for this target}}
void __attribute__((preserve_most)) funcL() {} // expected-error {{'preserve_most' calling convention is not supported for this target}}
void __attribute__((preserve_all)) funcM() {} // expected-error {{'preserve_all' calling convention is not supported for this target}}
void __attribute__((preserve_none)) funcN() {} // expected-error {{'preserve_none' calling convention is not supported for this target}}
void __attribute__((stdcall)) funcD() {} // expected-error {{'stdcall' calling convention is not supported for this target}}
void __attribute__((fastcall)) funcE() {} // expected-error {{'fastcall' calling convention is not supported for this target}}
void __attribute__((thiscall)) funcF() {} // expected-error {{'thiscall' calling convention is not supported for this target}}
#else

void __vectorcall func_vc() {}
void __regcall func_rc() {}
void __attribute__((vectorcall)) funcA() {}
void __attribute__((regcall)) funcB() {}
void __attribute__((ms_abi)) funcH() {}
void __attribute__((intel_ocl_bicc)) funcJ() {}
void __attribute__((swiftcall)) funcK() {}
void __attribute__((swiftasynccall)) funcKK() {}
void __attribute__((preserve_most)) funcL() {}
void __attribute__((preserve_all)) funcM() {}
void __attribute__((preserve_none)) funcN() {}

// Same function with different calling conventions. Error with a note pointing to the last decl.
void __attribute__((preserve_all)) funcR(); // expected-note {{previous declaration is here}}
void __attribute__((preserve_most)) funcR(); // expected-error {{function declared 'preserve_most' here was previously declared 'preserve_all'}}

void __attribute__((pascal)) funcG() {} // expected-warning {{'pascal' calling convention is not supported for this target}}

void __attribute__((stdcall)) funcD() {} // expected-warning {{'stdcall' calling convention is not supported for this target}}
void __attribute__((fastcall)) funcE() {} // expected-warning {{'fastcall' calling convention is not supported for this target}}
void __attribute__((thiscall)) funcF() {} // expected-warning {{'thiscall' calling convention is not supported for this target}}
#endif

void __attribute__((sysv_abi)) funcI() {}
void __attribute__((cdecl)) funcC() {}
