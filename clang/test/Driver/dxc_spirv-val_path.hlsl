// RUN: mkdir -p %t.dir

// RUN: echo "spirv-val" > %t.dir/spirv-val && chmod 754 %t.dir/spirv-val && %clang_dxc -spirv --spirv-val-path=%t.dir %s -Tlib_6_3 -Fo %t.dir/a.spv -### 2>&1 | FileCheck %s --check-prefix=SPIRV_VAL_PATH
// SPIRV_VAL_PATH:spirv-val{{(.exe)?}}" "--target-env" "vulkan1.3" "--scalar-block-layout" "{{.*}}.spv"

// RUN: echo "spirv-val" > %t.dir/spirv-val && chmod 754 %t.dir/spirv-val && %clang_dxc -spirv --spirv-val-path=%t.dir %s -Tlib_6_3 -fspv-target-env=vulkan1.2 -Fo %t.dir/a.spv -### 2>&1 | FileCheck %s --check-prefix=SPIRV_VAL_VK12
// SPIRV_VAL_VK12:spirv-val{{(.exe)?}}" "--target-env" "vulkan1.2" "--scalar-block-layout" "{{.*}}.spv"

// RUN: %clang_dxc -spirv -Tlib_6_3 -ccc-print-bindings --spirv-val-path=%t.dir -Fo %t.spv  %s 2>&1 | FileCheck %s --check-prefix=BINDINGS
// BINDINGS: "spirv1.6-unknown-vulkan1.3-library" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[spv:.+]].spv"
// BINDINGS-NEXT: "spirv1.6-unknown-vulkan1.3-library" - "hlsl::Validator", inputs: ["[[spv]].spv"], output: "{{.+}}.obj"

// RUN: %clang_dxc -spirv -Tlib_6_3 -ccc-print-phases --spirv-val-path=%t.dir -Fo %t.spv  %s 2>&1 | FileCheck %s --check-prefix=PHASES

// PHASES: 0: input, "[[INPUT:.+]]", hlsl
// PHASES-NEXT: 1: preprocessor, {0}, c++-cpp-output
// PHASES-NEXT: 2: compiler, {1}, ir
// PHASES-NEXT: 3: backend, {2}, assembler
// PHASES-NEXT: 4: assembler, {3}, object
// PHASES-NEXT: 5: binary-analyzer, {4}, object
