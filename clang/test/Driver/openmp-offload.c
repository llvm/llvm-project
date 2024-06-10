///
/// Perform several driver tests for OpenMP offloading
///

/// ###########################################################################

/// Check whether an invalid OpenMP target is specified:
// RUN:   not %clang -### -fopenmp=libomp -fopenmp-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check warning for empty -fopenmp-targets
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// CHK-EMPTY-OMPTARGETS: warning: joined argument expects additional value: '-fopenmp-targets='

/// ###########################################################################

/// Check error for no -fopenmp option
// RUN:   not %clang -### -fopenmp-targets=powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FOPENMP %s
// RUN:   not %clang -### -fopenmp=libgomp -fopenmp-targets=powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FOPENMP %s
// CHK-NO-FOPENMP: error: '-fopenmp-targets' must be used in conjunction with a '-fopenmp' option compatible with offloading; e.g., '-fopenmp=libomp' or '-fopenmp=libiomp5'

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu,powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: OpenMP offloading target 'powerpc64le-ibm-linux-gnu' is similar to target 'powerpc64le-ibm-linux-gnu' already specified; will be ignored

/// ###########################################################################

/// Check -Xopenmp-target=powerpc64le-ibm-linux-gnu -mcpu=pwr7 is passed when compiling for the device.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu -Xopenmp-target=powerpc64le-ibm-linux-gnu -mcpu=pwr7 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-EQ-TARGET %s

// CHK-FOPENMP-EQ-TARGET: clang{{.*}} "-target-cpu" "pwr7" {{.*}}"-fopenmp-is-target-device"

/// ###########################################################################

/// Check -Xopenmp-target -mcpu=pwr7 is passed when compiling for the device.
// RUN:   %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu -Xopenmp-target -mcpu=pwr7 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET %s

// CHK-FOPENMP-TARGET: clang{{.*}} "-target-cpu" "pwr7" {{.*}}"-fopenmp-is-target-device"

/// ##########################################################################

/// Check -mcpu=pwr7 is passed to the same triple.
// RUN:    %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu --target=powerpc64le-ibm-linux-gnu -mcpu=pwr7 %s 2>&1 \
// RUN:    | FileCheck -check-prefix=CHK-FOPENMP-MCPU-TO-SAME-TRIPLE %s

// CHK-FOPENMP-MCPU-TO-SAME-TRIPLE: clang{{.*}} "-target-cpu" "pwr7" {{.*}}"-fopenmp-is-target-device"

/// ##########################################################################

/// Check -march=pwr7 is NOT passed to nvptx64-nvidia-cuda.
// RUN:    not %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --target=powerpc64le-ibm-linux-gnu -march=pwr7 %s 2>&1 \
// RUN:    | FileCheck -check-prefix=CHK-FOPENMP-MARCH-TO-GPU %s

// CHK-FOPENMP-MARCH-TO-GPU-NOT: clang{{.*}} "-target-cpu" "pwr7" {{.*}}"-fopenmp-is-target-device"

/// ###########################################################################

/// Check -march=pwr7 is NOT passed to x86_64-unknown-linux-gnu.
// RUN:    not %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=x86_64-unknown-linux-gnu --target=powerpc64le-ibm-linux-gnu -march=pwr7 %s 2>&1 \
// RUN:    | FileCheck -check-prefix=CHK-FOPENMP-MARCH-TO-X86 %s

// CHK-FOPENMP-MARCH-TO-X86-NOT: clang{{.*}} "-target-cpu" "pwr7" {{.*}}"-fopenmp-is-target-device"

/// ###########################################################################

/// Check -Xopenmp-target triggers error when multiple triples are used.
// RUN:   not %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu,powerpc64le-unknown-linux-gnu -Xopenmp-target -mcpu=pwr8 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-AMBIGUOUS-ERROR %s

// CHK-FOPENMP-TARGET-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for -Xopenmp-target, specify triple using -Xopenmp-target=<triple>

/// ###########################################################################

/// Check -Xopenmp-target triggers error when an option requiring arguments is passed to it.
// RUN:   not %clang -### -no-canonical-prefixes -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu -Xopenmp-target -Xopenmp-target -mcpu=pwr8 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FOPENMP-TARGET-NESTED-ERROR %s

// CHK-FOPENMP-TARGET-NESTED-ERROR: clang{{.*}} error: invalid -Xopenmp-target argument: '-Xopenmp-target -Xopenmp-target', options requiring arguments are unsupported

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.
// RUN: %clang -ccc-print-phases -fopenmp=libomp --target=powerpc64-ibm-linux-gnu \
// RUN:   -fopenmp-targets=powerpc64-ibm-linux-gnu %s 2>&1 | FileCheck -check-prefix=CHK-PHASES %s
//      CHK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHK-PHASES-NEXT: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHK-PHASES-NEXT: 2: compiler, {1}, ir, (host-openmp)
// CHK-PHASES-NEXT: 3: input, "[[INPUT]]", c, (device-openmp)
// CHK-PHASES-NEXT: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHK-PHASES-NEXT: 5: compiler, {4}, ir, (device-openmp)
// CHK-PHASES-NEXT: 6: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {2}, "device-openmp (powerpc64-ibm-linux-gnu)" {5}, ir
// CHK-PHASES-NEXT: 7: backend, {6}, assembler, (device-openmp)
// CHK-PHASES-NEXT: 8: assembler, {7}, object, (device-openmp)
// CHK-PHASES-NEXT: 9: offload, "device-openmp (powerpc64-ibm-linux-gnu)" {8}, object
// CHK-PHASES-NEXT: 10: clang-offload-packager, {9}, image, (device-openmp)
// CHK-PHASES-NEXT: 11: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {2}, "device-openmp (powerpc64-ibm-linux-gnu)" {10}, ir
// CHK-PHASES-NEXT: 12: backend, {11}, assembler, (host-openmp)
// CHK-PHASES-NEXT: 13: assembler, {12}, object, (host-openmp)
// CHK-PHASES-NEXT: 14: clang-linker-wrapper, {13}, image, (host-openmp)

/// ###########################################################################

/// Check the phases when using multiple targets and multiple source files
// RUN: %clang -ccc-print-phases -lsomelib -fopenmp=libomp --target=powerpc64-ibm-linux-gnu \
// RUN:   -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64-ibm-linux-gnu %s %s 2>&1 | FileCheck -check-prefix=CHK-PHASES-FILES %s
//      CHK-PHASES-FILES: 0: input, "somelib", object, (host-openmp)
// CHK-PHASES-FILES-NEXT: 1: input, "[[INPUT:.+]]", c, (host-openmp)
// CHK-PHASES-FILES-NEXT: 2: preprocessor, {1}, cpp-output, (host-openmp)
// CHK-PHASES-FILES-NEXT: 3: compiler, {2}, ir, (host-openmp)
// CHK-PHASES-FILES-NEXT: 4: input, "[[INPUT]]", c, (device-openmp)
// CHK-PHASES-FILES-NEXT: 5: preprocessor, {4}, cpp-output, (device-openmp)
// CHK-PHASES-FILES-NEXT: 6: compiler, {5}, ir, (device-openmp)
// CHK-PHASES-FILES-NEXT: 7: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {3}, "device-openmp (powerpc64-ibm-linux-gnu)" {6}, ir
// CHK-PHASES-FILES-NEXT: 8: backend, {7}, assembler, (device-openmp)
// CHK-PHASES-FILES-NEXT: 9: assembler, {8}, object, (device-openmp)
// CHK-PHASES-FILES-NEXT: 10: offload, "device-openmp (powerpc64-ibm-linux-gnu)" {9}, object
// CHK-PHASES-FILES-NEXT: 11: input, "[[INPUT]]", c, (device-openmp)
// CHK-PHASES-FILES-NEXT: 12: preprocessor, {11}, cpp-output, (device-openmp)
// CHK-PHASES-FILES-NEXT: 13: compiler, {12}, ir, (device-openmp)
// CHK-PHASES-FILES-NEXT: 14: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {3}, "device-openmp (x86_64-pc-linux-gnu)" {13}, ir
// CHK-PHASES-FILES-NEXT: 15: backend, {14}, assembler, (device-openmp)
// CHK-PHASES-FILES-NEXT: 16: assembler, {15}, object, (device-openmp)
// CHK-PHASES-FILES-NEXT: 17: offload, "device-openmp (x86_64-pc-linux-gnu)" {16}, object
// CHK-PHASES-FILES-NEXT: 18: clang-offload-packager, {10, 17}, image, (device-openmp)
// CHK-PHASES-FILES-NEXT: 19: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {3}, "device-openmp (powerpc64-ibm-linux-gnu)" {18}, ir
// CHK-PHASES-FILES-NEXT: 20: backend, {19}, assembler, (host-openmp)
// CHK-PHASES-FILES-NEXT: 21: assembler, {20}, object, (host-openmp)
// CHK-PHASES-FILES-NEXT: 22: input, "[[INPUT]]", c, (host-openmp)
// CHK-PHASES-FILES-NEXT: 23: preprocessor, {22}, cpp-output, (host-openmp)
// CHK-PHASES-FILES-NEXT: 24: compiler, {23}, ir, (host-openmp)
// CHK-PHASES-FILES-NEXT: 25: input, "[[INPUT]]", c, (device-openmp)
// CHK-PHASES-FILES-NEXT: 26: preprocessor, {25}, cpp-output, (device-openmp)
// CHK-PHASES-FILES-NEXT: 27: compiler, {26}, ir, (device-openmp)
// CHK-PHASES-FILES-NEXT: 28: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {24}, "device-openmp (powerpc64-ibm-linux-gnu)" {27}, ir
// CHK-PHASES-FILES-NEXT: 29: backend, {28}, assembler, (device-openmp)
// CHK-PHASES-FILES-NEXT: 30: assembler, {29}, object, (device-openmp)
// CHK-PHASES-FILES-NEXT: 31: offload, "device-openmp (powerpc64-ibm-linux-gnu)" {30}, object
// CHK-PHASES-FILES-NEXT: 32: input, "[[INPUT]]", c, (device-openmp)
// CHK-PHASES-FILES-NEXT: 33: preprocessor, {32}, cpp-output, (device-openmp)
// CHK-PHASES-FILES-NEXT: 34: compiler, {33}, ir, (device-openmp)
// CHK-PHASES-FILES-NEXT: 35: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {24}, "device-openmp (x86_64-pc-linux-gnu)" {34}, ir
// CHK-PHASES-FILES-NEXT: 36: backend, {35}, assembler, (device-openmp)
// CHK-PHASES-FILES-NEXT: 37: assembler, {36}, object, (device-openmp)
// CHK-PHASES-FILES-NEXT: 38: offload, "device-openmp (x86_64-pc-linux-gnu)" {37}, object
// CHK-PHASES-FILES-NEXT: 39: clang-offload-packager, {31, 38}, image, (device-openmp)
// CHK-PHASES-FILES-NEXT: 40: offload, "host-openmp (powerpc64-ibm-linux-gnu)" {24}, "device-openmp (powerpc64-ibm-linux-gnu)" {39}, ir
// CHK-PHASES-FILES-NEXT: 41: backend, {40}, assembler, (host-openmp)
// CHK-PHASES-FILES-NEXT: 42: assembler, {41}, object, (host-openmp)
// CHK-PHASES-FILES-NEXT: 43: clang-linker-wrapper, {0, 21, 42}, image, (host-openmp)

/// Check -fopenmp-is-target-device is passed when compiling for the device.
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-fopenmp-is-target-device %s

// CHK-fopenmp-is-target-device: "-cc1"{{.*}} "-aux-triple" "powerpc64le-unknown-linux" {{.*}}"-fopenmp-is-target-device" "-fopenmp-host-ir-file-path" {{.*}}.c"

/// Check arguments to the linker wrapper
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NEW-DRIVER %s

// CHK-NEW-DRIVER: clang-linker-wrapper{{.*}}"--host-triple=powerpc64le-unknown-linux"{{.*}}--{{.*}}"-lomp"{{.*}}"-lomptarget"

/// Check arguments to the linker wrapper
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NEW-DRIVER-DEBUG %s

// CHK-NEW-DRIVER-DEBUG: clang-linker-wrapper{{.*}} "--device-debug"

/// Check arguments to the linker wrapper
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu \
// RUN:     -mllvm -abc %s 2>&1 | FileCheck -check-prefix=CHK-NEW-DRIVER-MLLVM %s

// CHK-NEW-DRIVER-MLLVM: clang-linker-wrapper{{.*}} "-abc"

//
// Ensure that we generate the correct bindings for '-fsyntax-only' for OpenMP.
//
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu \
// RUN:     -fsyntax-only -ccc-print-bindings %s 2>&1 | FileCheck -check-prefix=CHK-SYNTAX-ONLY %s
// CHK-SYNTAX-ONLY: # "powerpc64le-ibm-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: (nothing)
// CHK-SYNTAX-ONLY: # "powerpc64le-unknown-linux" - "clang", inputs: ["[[INPUT]]", (nothing)], output: (nothing)

//
// Ensure that we can generate the correct arguments for '-fsyntax-only' for
// OpenMP.
//
// RUN:   %clang -### --target=powerpc64le-linux -fopenmp=libomp -fopenmp-targets=powerpc64le-ibm-linux-gnu \
// RUN:     -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHK-SYNTAX-ONLY-ARGS %s
// CHK-SYNTAX-ONLY-ARGS: "-cc1" "-triple" "powerpc64le-ibm-linux-gnu"{{.*}}"-fsyntax-only"
// CHK-SYNTAX-ONLY-ARGS: "-cc1" "-triple" "powerpc64le-unknown-linux"{{.*}}"-fsyntax-only"
