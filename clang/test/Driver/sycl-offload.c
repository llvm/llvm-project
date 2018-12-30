///
/// Perform several driver tests for SYCL offloading
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   %clang -### -fsycl -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: SYCL target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check warning for empty -fsycl-targets
// RUN:   %clang -### -fsycl -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// CHK-EMPTY-SYCLTARGETS: warning: joined argument expects additional value: '-fsycl-targets='

/// ###########################################################################

/// Check error for no -fsycl option
// RUN:   %clang -### -fsycl-targets=spir64-unknown-linux-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: The option -fsycl-targets must be used in conjunction with -fsycl to enable offloading.

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir64-unknown-linux-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: The SYCL offloading target 'spir64-unknown-linux-sycldevice' is similar to target 'spir64-unknown-linux-sycldevice' already specified - will be ignored.

/// ###########################################################################

/// Check -Xsycl-target triggers error when multiple triples are used.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir-unknown-linux-sycldevice -Xsycl-target -mcpu=pentium4 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s

// CHK-FSYCL-TARGET-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for -Xsycl-target, specify triple using -Xsycl-target=<triple>

/// ###########################################################################

/// Check -Xsycl-target triggers error when an option requiring arguments is passed to it.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice -Xsycl-target -Xsycl-target -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-NESTED-ERROR %s

// CHK-FSYCL-TARGET-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target argument: '-Xsycl-target -Xsycl-target', options requiring arguments are unsupported

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-{{.*}}-sycldevice)" {4}, cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES: 11: backend, {10}, assembler, (device-sycl)
// CHK-PHASES: 12: assembler, {11}, object, (device-sycl)
// CHK-PHASES: 13: linker, {12}, image, (device-sycl)
// CHK-PHASES: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-PHASES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-linux-sycldevice)" {14}, image

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES-LIB: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-LIB: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-LIB: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-linux-sycldevice)" {5}, cpp-output
// CHK-PHASES-LIB: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-LIB: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-LIB: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-LIB: 10: linker, {0, 9}, image, (host-sycl)
// CHK-PHASES-LIB: 11: input, "somelib", object, (device-sycl)
// CHK-PHASES-LIB: 12: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LIB: 13: backend, {12}, assembler, (device-sycl)
// CHK-PHASES-LIB: 14: assembler, {13}, object, (device-sycl)
// CHK-PHASES-LIB: 15: linker, {11, 14}, image, (device-sycl)
// CHK-PHASES-LIB: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASES-LIB: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-linux-sycldevice)" {16}, image

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s

// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: input, "[[INPUT1]]", c, (device-sycl)
// CHK-PHASES-FILES: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-linux-sycldevice)" {5}, cpp-output
// CHK-PHASES-FILES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-FILES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-FILES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-FILES: 10: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 11: preprocessor, {10}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 12: input, "[[INPUT2]]", c, (device-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 14: compiler, {13}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-linux-sycldevice)" {14}, cpp-output
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (host-sycl)
// CHK-PHASES-FILES: 17: backend, {16}, assembler, (host-sycl)
// CHK-PHASES-FILES: 18: assembler, {17}, object, (host-sycl)
// CHK-PHASES-FILES: 19: linker, {0, 9, 18}, image, (host-sycl)
// CHK-PHASES-FILES: 20: input, "somelib", object, (device-sycl)
// CHK-PHASES-FILES: 21: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FILES: 22: backend, {21}, assembler, (device-sycl)
// CHK-PHASES-FILES: 23: assembler, {22}, object, (device-sycl)
// CHK-PHASES-FILES: 24: compiler, {13}, ir, (device-sycl)
// CHK-PHASES-FILES: 25: backend, {24}, assembler, (device-sycl)
// CHK-PHASES-FILES: 26: assembler, {25}, object, (device-sycl)
// CHK-PHASES-FILES: 27: linker, {20, 23, 26}, image, (device-sycl)
// CHK-PHASES-FILES: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASES-FILES: 29: offload, "host-sycl (x86_64-unknown-linux-gnu)" {19}, "device-sycl (spir64-unknown-linux-sycldevice)" {28}, image

/// ###########################################################################

/// Check separate compilation with offloading - bundling actions
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -c -o %t.o -lsomelib -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BUACTIONS %s
// CHK-BUACTIONS: 0: input, "[[INPUT:.+\.c]]", c, (device-sycl)
// CHK-BUACTIONS: 1: preprocessor, {0}, cpp-output, (device-sycl)
// CHK-BUACTIONS: 2: compiler, {1}, ir, (device-sycl)
// CHK-BUACTIONS: 3: backend, {2}, assembler, (device-sycl)
// CHK-BUACTIONS: 4: assembler, {3}, object, (device-sycl)
// CHK-BUACTIONS: 5: offload, "device-sycl (spir64-unknown-linux-sycldevice)" {4}, object
// CHK-BUACTIONS: 6: input, "[[INPUT]]", c, (host-sycl)
// CHK-BUACTIONS: 7: preprocessor, {6}, cpp-output, (host-sycl)
// CHK-BUACTIONS: 8: compiler, {1}, sycl-header, (device-sycl)
// CHK-BUACTIONS: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {7}, "device-sycl (spir64-unknown-linux-sycldevice)" {8}, cpp-output
// CHK-BUACTIONS: 10: compiler, {9}, ir, (host-sycl)
// CHK-BUACTIONS: 11: backend, {10}, assembler, (host-sycl)
// CHK-BUACTIONS: 12: assembler, {11}, object, (host-sycl)
// CHK-BUACTIONS: 13: clang-offload-bundler, {5, 12}, object, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -lsomelib -fsycl-targets=spir64-unknown-linux-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBACTIONS %s
// CHK-UBACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBACTIONS: 1: input, "[[INPUT:.+\.o]]", object, (host-sycl)
// CHK-UBACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBACTIONS: 3: linker, {0, 2}, image, (host-sycl)
// CHK-UBACTIONS: 4: input, "somelib", object, (device-sycl)
// CHK-UBACTIONS: 5: linker, {4, 2}, image, (device-sycl)
// CHK-UBACTIONS: 6: clang-offload-wrapper, {5}, object, (device-sycl)
// CHK-UBACTIONS: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-linux-sycldevice)" {6}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling with source
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl %t.o -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-UBUACTIONS: 4: preprocessor, {3}, cpp-output, (host-sycl)
// CHK-UBUACTIONS: 5: input, "[[INPUT2]]", c, (device-sycl)
// CHK-UBUACTIONS: 6: preprocessor, {5}, cpp-output, (device-sycl)
// CHK-UBUACTIONS: 7: compiler, {6}, sycl-header, (device-sycl)
// CHK-UBUACTIONS: 8: offload, "host-sycl (x86_64-unknown-linux-gnu)" {4}, "device-sycl (spir64-unknown-linux-sycldevice)" {7}, cpp-output
// CHK-UBUACTIONS: 9: compiler, {8}, ir, (host-sycl)
// CHK-UBUACTIONS: 10: backend, {9}, assembler, (host-sycl)
// CHK-UBUACTIONS: 11: assembler, {10}, object, (host-sycl)
// CHK-UBUACTIONS: 12: linker, {0, 2, 11}, image, (host-sycl)
// CHK-UBUACTIONS: 13: input, "somelib", object, (device-sycl)
// CHK-UBUACTIONS: 14: compiler, {6}, ir, (device-sycl)
// CHK-UBUACTIONS: 15: backend, {14}, assembler, (device-sycl)
// CHK-UBUACTIONS: 16: assembler, {15}, object, (device-sycl)
// CHK-UBUACTIONS: 17: linker, {13, 2, 16}, image, (device-sycl)
// CHK-UBUACTIONS: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-UBUACTIONS: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64-unknown-linux-sycldevice)" {18}, image

/// ###########################################################################

/// Check -fsycl-is-device is passed when compiling for the device.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}}.c
