/// Test -fsycl-link: the SYCL device code is linked and wrapped into a host
/// object by clang-linker-wrapper, without running the host linker.

// RUN: touch %t.a.o %t.b.o

/// clang-linker-wrapper is the only link job, and it gets --sycl-device-link.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   %t.a.o %t.b.o -o dev.o 2>&1 | FileCheck %s --check-prefix=LINK
// LINK: clang-linker-wrapper"
// LINK-SAME: "--sycl-device-link"
// LINK-SAME: "-o" "dev.o"

// RUN: %clang -ccc-print-bindings --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -fsycl-link %t.a.o %t.b.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BINDINGS
// BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["{{.*}}a.o", "{{.*}}b.o"], output: "dev.o"
// BINDINGS-NOT: {{.}}

/// The default output name is the same as for a host link.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   %t.a.o 2>&1 | FileCheck %s --check-prefix=DEFAULT-NAME
// DEFAULT-NAME: clang-linker-wrapper"{{.*}} "--sycl-device-link"{{.*}} "-o" "a.out"

/// Source inputs are compiled, then device linked.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link %s \
// RUN:   -o dev.o 2>&1 | FileCheck %s --check-prefix=SOURCE
// SOURCE: "-cc1"{{.*}} "-fsycl-is-device"
// SOURCE: "-cc1"{{.*}} "-fsycl-is-host"
// SOURCE: clang-linker-wrapper"{{.*}} "--sycl-device-link"{{.*}} "-o" "dev.o"

// RUN: %clang -### --target=x86_64-pc-windows-msvc -fsycl -fsycl-link \
// RUN:   %t.a.o -o dev.obj 2>&1 | FileCheck %s --check-prefix=WINDOWS
// WINDOWS: clang-linker-wrapper"{{.*}} "--sycl-device-link"{{.*}} "-out:dev.obj"
// RUN: %clang -ccc-print-bindings --target=x86_64-pc-windows-msvc -fsycl \
// RUN:   -fsycl-link %t.a.o -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WINDOWS-BINDINGS
// WINDOWS-BINDINGS: "x86_64-pc-windows-msvc{{.*}}" - "Offload::Linker", inputs: ["{{.*}}a.o"], output: "dev.obj"
// WINDOWS-BINDINGS-NOT: {{.}}

/// Without -fsycl-link, the host link is run as usual.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-SYCL-LINK
// NO-SYCL-LINK: clang-linker-wrapper"
// NO-SYCL-LINK-NOT: "--sycl-device-link"

/// -fsycl-link requires -fsycl.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl-link %t.a.o \
// RUN:   2>&1 | FileCheck %s --check-prefix=NO-FSYCL
// NO-FSYCL: error: invalid argument '-fsycl-link' only allowed with '-fsycl'

/// -fsycl-link needs both the host and the device compilation.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   --offload-device-only %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLY -DOPT=--offload-device-only
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   -fsycl-device-only %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLY -DOPT=-fsycl-device-only
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   --offload-host-only %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLY -DOPT=--offload-host-only
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   -fsycl-host-only %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLY -DOPT=-fsycl-host-only
// ONLY: error: invalid argument '-fsycl-link' not allowed with '[[OPT]]'

/// -fsycl-link is unused when there is no link step.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link -c %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNUSED \
// RUN:   --implicit-check-not=--sycl-device-link
// UNUSED: warning: argument unused during compilation: '-fsycl-link'
