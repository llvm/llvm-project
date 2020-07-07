// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=AMX-TILE %s

// AMX-TILE: #define __AMXTILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-bf16 -x c -E -dM -o - %s | FileCheck -check-prefix=AMX-BF16 %s

// AMX-BF16: #define __AMXBF16__ 1
// AMX-BF16: #define __AMXTILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mamx-int8 -x c -E -dM -o - %s | FileCheck -check-prefix=AMX-INT8 %s

// AMX-INT8: #define __AMXINT8__ 1
// AMX-INT8: #define __AMXTILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-TILE %s

// NOAMX-TILE-NOT: #define __AMXTILE__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-bf16 -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-BF16 %s

// NOAMX-BF16-NOT: #define __AMXBF16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -amx-bf16 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-BF16 %s

// NOAMX-BF16-NOT: #define __AMXTILE__ 1
// NOAMX-BF16-NOT: #define __AMXBF16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-amx-int8 -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-INT8 %s

// NOAMX-INT8-NOT: #define __AMXINT8__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -amx-int8 -mno-amx-tile -x c -E -dM -o - %s | FileCheck  -check-prefix=NOAMX-INT8 %s

// NOAMX-INT8-NOT: #define __AMXTILE__ 1
// NOAMX-INT8-NOT: #define __AMXINT8__ 1
