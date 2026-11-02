// RUN: %clang_dxc -T vs_6_0 -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_dxc -T vs_6_0 -pack-prefix-stable -### %s 2>&1 | FileCheck %s --check-prefix=STABLE
// RUN: %clang_dxc -T vs_6_0 -pack-optimized -### %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZED
// RUN: %clang_dxc -T vs_6_0 /pack-prefix-stable -### %s 2>&1 | FileCheck %s --check-prefix=STABLE
// RUN: %clang_dxc -T vs_6_0 /pack-optimized -### %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZED
// RUN: not %clang_dxc -T vs_6_0 -pack-prefix-stable -pack-optimized -### %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not %clang_dxc -T vs_6_0 -pack-optimized -pack-prefix-stable -### %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: %clang --target=dxil-pc-shadermodel6.0-vertex -fdx-semantic-signature-packing-mode=prefix-stable -### %s 2>&1 | FileCheck %s --check-prefix=STABLE
// RUN: %clang --target=dxil-pc-shadermodel6.0-vertex -fdx-semantic-signature-packing-mode=optimized -### %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZED
// RUN: %clang --target=dxil-pc-shadermodel6.0-vertex -fdx-semantic-signature-packing-mode=prefix-stable -fdx-semantic-signature-packing-mode=optimized -### %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZED
// RUN: not %clang --target=dxil-pc-shadermodel6.0-vertex -fdx-semantic-signature-packing-mode=invalid -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang --target=dxil-pc-shadermodel6.0-vertex -fdx-semantic-signature-packing-mode= -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not %clang -x c -Xclang -fdx-semantic-signature-packing-mode=optimized -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=NOT-HLSL

// DEFAULT-NOT: -fdx-semantic-signature-packing-mode=
// STABLE: "-fdx-semantic-signature-packing-mode=prefix-stable"
// STABLE-NOT: "-fdx-semantic-signature-packing-mode=optimized"
// OPTIMIZED-NOT: "-fdx-semantic-signature-packing-mode=prefix-stable"
// OPTIMIZED: "-fdx-semantic-signature-packing-mode=optimized"
// CONFLICT: error: invalid argument '-pack-prefix-stable' not allowed with '-pack-optimized'
// INVALID: error: invalid value 'invalid' in '-fdx-semantic-signature-packing-mode=invalid'
// EMPTY: error: invalid value '' in '-fdx-semantic-signature-packing-mode='
// NOT-HLSL: error: invalid argument '-fdx-semantic-signature-packing-mode' not allowed with 'C'
