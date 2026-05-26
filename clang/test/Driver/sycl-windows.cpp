/// Windows SYCL runtime library linking tests
//
// NOTE: SYCL runtime library dependency is added at compiler stage via
// --dependent-lib (embedded in object file), similar to CRT libraries.
// Tests check for --dependent-lib in compiler (-cc1) output and
// -libpath: in linker output.

/// Test 1: /MD (explicit) and release library dependency
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MD %s
// CHECK-MD: "-cc1"
// CHECK-MD: "--dependent-lib=LLVMSYCL"
// CHECK-MD: clang-linker-wrapper"
// CHECK-MD-SAME: "-libpath:{{.*}}{{[/\\]+}}lib"

/// Test 2: /MT is rejected with clear error
// RUN: not %clang_cl -fsycl /MT --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MT-ERROR %s
// CHECK-MT-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 3: /MTd is also rejected
// RUN: not %clang_cl -fsycl /MTd --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MTD-ERROR %s
// CHECK-MTD-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 4: /MD uses release library
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RELEASE %s
// CHECK-RELEASE: "-cc1"
// CHECK-RELEASE: "--dependent-lib=LLVMSYCL"
// CHECK-RELEASE-NOT: LLVMSYCLd

/// Test 5: /MDd uses debug library
// RUN: %clang_cl -### -fsycl /MDd --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG: "-cc1"
// CHECK-DEBUG: "--dependent-lib=LLVMSYCLd"
// CHECK-DEBUG-NOT: "--dependent-lib=LLVMSYCL"

/// Test 6: -fms-runtime-lib=static is rejected
// RUN: not %clang_cl -fsycl -fms-runtime-lib=static \
// RUN:   --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-STATIC-ERROR %s
// CHECK-STATIC-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 7: -fms-runtime-lib=dll_dbg uses debug library
// RUN: %clang_cl -### -fsycl -fms-runtime-lib=dll_dbg \
// RUN:   --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FLAG-DEBUG %s
// CHECK-FLAG-DEBUG: "-cc1"
// CHECK-FLAG-DEBUG: "--dependent-lib=LLVMSYCLd"

/// Test 8: LNK4078 warning is suppressed
// RUN: %clang_cl -### -fsycl --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-IGNORE %s
// CHECK-IGNORE: clang-linker-wrapper"
// CHECK-IGNORE: "/IGNORE:4078"

/// Test 9: -nolibsycl skips library dependency and CRT check
// RUN: %clang_cl -### -fsycl -nolibsycl /MT \
// RUN:   --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOLIBSYCL %s
// CHECK-NOLIBSYCL-NOT: error:
// CHECK-NOLIBSYCL-NOT: "--dependent-lib=LLVMSYCL"

/// Test 10: Explicit /MD results in correct library
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EXPLICIT-MD %s
// CHECK-EXPLICIT-MD: "-cc1"
// CHECK-EXPLICIT-MD: "--dependent-lib=LLVMSYCL"

/// Test 11: Library search path is added at linker stage
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LIBPATH %s
// CHECK-LIBPATH: clang-linker-wrapper"
// CHECK-LIBPATH: "-libpath:{{.*}}{{[/\\]+}}lib"

/// Test 12: clang (non-clang-cl) with MSVC target uses -defaultlib:
// RUN: %clang -### -fsycl -fms-runtime-lib=dll --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-CLANG-DEFAULTLIB %s
// CHECK-CLANG-DEFAULTLIB: clang-linker-wrapper"
// CHECK-CLANG-DEFAULTLIB: "-libpath:{{.*}}{{[/\\]+}}lib"
// CHECK-CLANG-DEFAULTLIB: "-defaultlib:LLVMSYCL"

/// Test 13: clang with -fms-runtime-lib=dll_dbg uses debug library via -defaultlib:
// RUN: %clang -### -fsycl -fms-runtime-lib=dll_dbg --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-CLANG-DEBUG %s
// CHECK-CLANG-DEBUG: clang-linker-wrapper"
// CHECK-CLANG-DEBUG: "-defaultlib:LLVMSYCLd"

/// Test 14: Default CRT behavior - release library when no CRT flag specified
// RUN: %clang -### -fsycl -fms-runtime-lib=dll --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT-CRT %s
// CHECK-DEFAULT-CRT: "-cc1"
// CHECK-DEFAULT-CRT: "--dependent-lib=LLVMSYCL"
// CHECK-DEFAULT-CRT-NOT: LLVMSYCLd

/// Test 15: Separate compilation - compile step embeds --dependent-lib in object
// RUN: %clang -### -fsycl -c -fms-runtime-lib=dll --target=x86_64-pc-windows-msvc -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SEP-COMPILE %s
// CHECK-SEP-COMPILE: "--dependent-lib=LLVMSYCL"

/// Test 16: Separate compilation - link step adds -libpath: and -defaultlib: for pre-compiled object
// RUN: touch %t.obj
// RUN: %clang -### -fsycl -fms-runtime-lib=dll --target=x86_64-pc-windows-msvc -- %t.obj 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SEP-LINK %s
// CHECK-SEP-LINK: "-libpath:{{.*}}{{[/\\]+}}lib"
// CHECK-SEP-LINK: "-defaultlib:LLVMSYCL"
