// Test the behavior of -target-cpu for cc1 with amdgpu subarches. -target-cpu
// usage is an edge case, typical invocations should use a subarch in the triple
// and omit the -target-cpu argument. If -target-cpu is used, it should still be
// respected (particularly in the case where the triple is a major subarch
// covering the -target-cpu).

// Specific subarch, no -target-cpu: GPU implied by the subarch.
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -E -dM %s 2>&1 | \
// RUN:   FileCheck --check-prefix=GFX90A %s
// GFX90A-DAG: #define __amdgcn_processor__ "gfx90a"
// GFX90A-DAG: #define __gfx90a__ 1
// GFX90A-DAG: #define __GFX9__ 1

// Generic-family subarch, no -target-cpu: GPU is the generic target.
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -E -dM %s 2>&1 | \
// RUN:   FileCheck --check-prefix=GFX9-GENERIC %s
// GFX9-GENERIC-DAG: #define __amdgcn_processor__ "gfx9_generic"
// GFX9-GENERIC-DAG: #define __gfx9_generic__ 1

// An explicit -target-cpu overrides the subarch (manual cc1 invocation).
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -target-cpu gfx900 -E -dM %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OVERRIDE %s
// OVERRIDE-DAG: #define __amdgcn_processor__ "gfx900"
// OVERRIDE-DAG: #define __gfx900__ 1
