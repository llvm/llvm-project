// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple spirv64-unknown-unknown -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv32-unknown-unknown -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv64-intel -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} generic
// CHECK-SAME: {{^}}, igca_

// Triples that don't accept -target-cpu shouldn't list any valid values.
// RUN: not %clang_cc1 -triple spirv64-unknown-vulkan1.3 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: not %clang_cc1 -triple spirv64-unknown-chipstar -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NONE
// NONE: error: unknown target CPU 'not-a-cpu'
// NONE-NOT: note: valid target CPU values

// The AMDGCN flavor of SPIR-V lists AMDGPU CPUs instead.
// RUN: not %clang_cc1 -triple spirv64-amd-amdhsa -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=AMD
// AMD: error: unknown target CPU 'not-a-cpu'
// AMD-NEXT: note: valid target CPU values are:
// AMD-SAME: gfx906
// AMD-NOT: igca_
