// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix ASAN %s

// The address_safety attribute should be attached to functions
// when AddressSanitizer is enabled, unless no_address_safety_analysis attribute
// is present.

// CHECK-NOT:  NoAddressSafety1{{.*}} address_safety
// ASAN-NOT:  NoAddressSafety1{{.*}} address_safety
__attribute__((no_address_safety_analysis))
int NoAddressSafety1(int *a) { return *a; }

// CHECK-NOT:  NoAddressSafety2{{.*}} address_safety
// ASAN-NOT:  NoAddressSafety2{{.*}} address_safety
__attribute__((no_address_safety_analysis))
int NoAddressSafety2(int *a);
int NoAddressSafety2(int *a) { return *a; }

// CHECK-NOT:  AddressSafetyOk{{.*}} address_safety
// ASAN: AddressSafetyOk{{.*}} address_safety
int AddressSafetyOk(int *a) { return *a; }

// CHECK-NOT:  TemplateNoAddressSafety{{.*}} address_safety
// ASAN-NOT: TemplateNoAddressSafety{{.*}} address_safety
template<int i>
__attribute__((no_address_safety_analysis))
int TemplateNoAddressSafety() { return i; }

// CHECK-NOT:  TemplateAddressSafetyOk{{.*}} address_safety
// ASAN: TemplateAddressSafetyOk{{.*}} address_safety
template<int i>
int TemplateAddressSafetyOk() { return i; }

int force_instance = TemplateAddressSafetyOk<42>()
                   + TemplateNoAddressSafety<42>();

// Check that __cxx_global_var_init* get the address_safety attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// CHECK-NOT: @__cxx_global_var_init{{.*}}address_safety
// ASAN: @__cxx_global_var_init{{.*}}address_safety
