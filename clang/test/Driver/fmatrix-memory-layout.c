// RUN: %clang --target=x86_64-linux-gnu -fenable-matrix -fmatrix-memory-layout=column-major %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COL-MAJOR
// CHECK-COL-MAJOR:  -fenable-matrix
// CHECK-COL-MAJOR:  -mllvm
// CHECK-COL-MAJOR:  -enable-matrix
// CHECK-COL-MAJOR:  -fmatrix-memory-layout=column-major
// CHECK-COL-MAJOR:  -mllvm
// CHECK-COL-MAJOR:  -matrix-default-layout=column-major

// RUN: %clang --target=x86_64-linux-gnu -fenable-matrix -fmatrix-memory-layout=row-major %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ROW-MAJOR
// CHECK-ROW-MAJOR:  -fenable-matrix
// CHECK-ROW-MAJOR:  -mllvm
// CHECK-ROW-MAJOR:  -enable-matrix
// CHECK-ROW-MAJOR:  -fmatrix-memory-layout=row-major
// CHECK-ROW-MAJOR:  -mllvm
// CHECK-ROW-MAJOR:  -matrix-default-layout=row-major

// RUN: not %clang --target=x86_64-linux-gnu -fenable-matrix -fmatrix-memory-layout=error-major %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-MAJOR
// CHECK-ERROR-MAJOR: error: invalid value 'error-major' in '-fmatrix-memory-layout=error-major'

// RUN: %clang --target=x86_64-linux-gnu -fmatrix-memory-layout=column-major %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-COL-MAJOR-DISABLED
// CHECK-COL-MAJOR-DISABLED: clang: warning: argument unused during compilation: '-fmatrix-memory-layout=column-major'
// CHECK-COL-MAJOR-DISABLED-NOT:  -fenable-matrix
// CHECK-COL-MAJOR-DISABLED-NOT:  -mllvm
// CHECK-COL-MAJOR-DISABLED-NOT:  -enable-matrix
// CHECK-COL-MAJOR-DISABLED-NOT:  -fmatrix-memory-layout=column-major
// CHECK-COL-MAJOR-DISABLED-NOT:  -mllvm
// CHECK-COL-MAJOR-DISABLED-NOT:  -matrix-default-layout=column-major

// RUN: %clang --target=x86_64-linux-gnu -fmatrix-memory-layout=row-major %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-ROW-MAJOR-DISABLED
// CHECK-ROW-MAJOR-DISABLED: clang: warning: argument unused during compilation: '-fmatrix-memory-layout=row-major'
// CHECK-ROW-MAJOR-DISABLED-NOT:  -fenable-matrix
// CHECK-ROW-MAJOR-DISABLED-NOT:  -mllvm
// CHECK-ROW-MAJOR-DISABLED-NOT:  -enable-matrix
// CHECK-ROW-MAJOR-DISABLED-NOT:  -fmatrix-memory-layout=row-major
// CHECK-ROW-MAJOR-DISABLED-NOT:  -mllvm
// CHECK-ROW-MAJOR-DISABLED-NOT:  -matrix-default-layout=row-major

// RUN: %clang --target=x86_64-linux-gnu -fenable-matrix  %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MATRIX-ENABLED
// CHECK-MATRIX-ENABLED:  -fenable-matrix
// CHECK-MATRIX-ENABLED:  -mllvm
// CHECK-MATRIX-ENABLED:  -enable-matrix
// CHECK-MATRIX-ENABLED-NOT:  -fmatrix-memory-layout=row-major
// CHECK-MATRIX-ENABLED-NOT:  -fmatrix-memory-layout=column-major
// CHECK-MATRIX-ENABLED-NOT:  -mllvm
// CHECK-MATRIX-ENABLED-NOT:  -matrix-default-layout=row-major
// CHECK-MATRIX-ENABLED-NOT:  -matrix-default-layout=column-major

// RUN: not %clang --target=x86_64-linux-gnu -fenable-matrix -fmatrix-memory-layout=column-major -mllvm -matrix-default-layout=row-major %s -fsyntax-only 2>&1 | FileCheck %s --check-prefix=CHECK-MISMATCH-MAJOR
// CHECK-MISMATCH-MAJOR: error: -fmatrix-memory-layout=column-major conflicts with -mllvm -matrix-default-layout=row-major
