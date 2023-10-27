// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=AMX-TILE %s

// AMX-TILE: #define __AMX_TILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-bf16 -x c -E -dM -o - %s | FileCheck -check-prefix=AMX-BF16 %s

// AMX-BF16: #define __AMX_BF16__ 1
// AMX-BF16: #define __AMX_TILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-int8 -x c -E -dM -o - %s | FileCheck -check-prefix=AMX-INT8 %s

// AMX-INT8: #define __AMX_INT8__ 1
// AMX-INT8: #define __AMX_TILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-TILE %s

// NOAMX-TILE-NOT: #define __AMX_TILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-bf16 -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-BF16 %s

// NOAMX-BF16-NOT: #define __AMX_BF16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-bf16 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-BF16 %s

// NOAMX-BF16-NOT: #define __AMX_TILE__ 1
// NOAMX-BF16-NOT: #define __AMX_BF16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-int8 -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-INT8 %s

// NOAMX-INT8-NOT: #define __AMX_INT8__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-int8 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-INT8 %s

// NOAMX-INT8-NOT: #define __AMX_TILE__ 1
// NOAMX-INT8-NOT: #define __AMX_INT8__ 1
