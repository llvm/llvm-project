// RUN: %clang_dxc -T lib_6_7 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_dxc -fcgl -T lib_6_7 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=FCGL
// RUN: %clang_dxc  -T lib_6_7 foo.hlsl -Fo foo.dxc -### %s 2>&1 | FileCheck %s --check-prefix=EMITOBJ


// Make sure default use "-" as output and not emit obj.
// DEFAULT-NOT:"-emit-obj"
// DEFAULT:"-o" "-"

// Make sure -fcgl without -Fo use "-" as output.
// FCGL:"-o" "-"


// Make sure emit-obj when set -Fo.
// EMITOBJ:"-emit-obj"
