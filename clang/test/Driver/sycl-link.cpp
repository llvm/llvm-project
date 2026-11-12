/// Test -fsycl-link: the SYCL device code is linked and wrapped into a host
/// object by clang-linker-wrapper, without running the host linker.

// RUN: touch %t.a.o %t.b.o

/// clang-linker-wrapper is the only link job, and it gets --sycl-device-link
/// with the sysroot, library paths and link inputs instead of the host link
/// command.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree %t.a.o -L/foo \
// RUN:   -Wl,--whole-archive -lbar -Wl,--no-whole-archive %t.b.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=crt1.o \
// RUN:     --implicit-check-not='"-lc"'
// LINK: clang-linker-wrapper"
// LINK-SAME: "--sycl-device-link" "-o" "dev.o"
// LINK-SAME: "--sysroot={{[^"]*}}basic_linux_tree" "-L/foo"
// LINK-SAME: "-L{{[^"]*}}basic_linux_tree{{[^"]*}}{{[/\\]+}}usr{{[/\\]+}}lib"
// LINK-SAME: "{{[^"]*}}a.o" "--whole-archive" "-lbar" "--no-whole-archive" "{{[^"]*}}b.o"{{$}}

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
// WINDOWS: clang-linker-wrapper"{{.*}} "--sycl-device-link" "-o" "dev.obj" "{{[^"]*}}a.o"{{$}}
// RUN: %clang -ccc-print-bindings --target=x86_64-pc-windows-msvc -fsycl \
// RUN:   -fsycl-link %t.a.o -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WINDOWS-BINDINGS
// WINDOWS-BINDINGS: "x86_64-pc-windows-msvc{{.*}}" - "Offload::Linker", inputs: ["{{.*}}a.o"], output: "dev.obj"
// WINDOWS-BINDINGS-NOT: {{.}}

/// MSVC style links search for named inputs in -libpath: paths, and clang-cl
/// /link values are passed on after -o, so a /link /out: names the output as in
/// a host link.
// RUN: %clang -### --target=x86_64-pc-windows-msvc -fsycl -fsycl-link \
// RUN:   %t.a.o -L/foo -Wl,/wholearchive:bar.lib -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WINDOWS-LIBPATH
// WINDOWS-LIBPATH: clang-linker-wrapper"{{.*}} "--sycl-device-link" "-o" "dev.obj"
// WINDOWS-LIBPATH-SAME: "-libpath:/foo" "{{[^"]*}}a.o" "/wholearchive:bar.lib"{{$}}
// The input is given with /Tp, because clang-cl would read a path under /Users
// (common on macOS) as the /U option.
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl -fsycl-link \
// RUN:   /Tp%s /link /wholearchive:bar.lib /libpath:dir /out:dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CL-LINK
// CL-LINK: clang-linker-wrapper"{{.*}} "--sycl-device-link" "-o"
// CL-LINK-SAME: "{{[^"]*}}.obj" "/wholearchive:bar.lib" "/libpath:dir" "/out:dev.obj"{{$}}

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

/// The last of the mode options selects the mode.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   -fsycl-device-only -fsycl-host-only %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLY -DOPT=-fsycl-host-only
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link \
// RUN:   --offload-device-only --offload-host-device %t.a.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HOST-DEVICE
// HOST-DEVICE-NOT: error:
// HOST-DEVICE: clang-linker-wrapper"{{.*}} "--sycl-device-link"

/// -fsycl-link is unused when there is no link step.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-link -c %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNUSED \
// RUN:   --implicit-check-not=--sycl-device-link
// UNUSED: warning: argument unused during compilation: '-fsycl-link'
