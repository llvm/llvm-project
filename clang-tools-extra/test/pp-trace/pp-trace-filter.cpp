// RUN: pp-trace -callbacks 'File*,Macro*,-MacroUndefined' %s -- | FileCheck %s
// RUN: pp-trace -callbacks ' File* , Macro* , -MacroUndefined ' %s -- | FileCheck %s
// RUN: not pp-trace -callbacks '[' %s -- 2>&1 | FileCheck --check-prefix=INVALID %s

#define M 1
int i = M;
#undef M

// CHECK:      ---
// CHECK:      - Callback: FileChanged
// CHECK:      - Callback: MacroDefined
// CHECK:      - Callback: MacroExpands
// CHECK-NOT:  - Callback: MacroUndefined
// CHECK-NOT:  - Callback: EndOfMainFile
// CHECK:      ...

// INVALID: error: invalid glob pattern: [
