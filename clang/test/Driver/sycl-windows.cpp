/// Windows SYCL runtime library linking tests


/// Test 1: Auto-/MD is added when no CRT specified
// RUN: %clang_cl -### -fsycl --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AUTO-MD %s
// CHECK-AUTO-MD: "/MD"
// CHECK-AUTO-MD: "-defaultlib:LLVMSYCL.lib"

/// Test 2: /MT is rejected with clear error
// RUN: not %clang_cl -fsycl /MT --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MT-ERROR %s
// CHECK-MT-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 3: /MTd is also rejected
// RUN: not %clang_cl -fsycl /MTd --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MTD-ERROR %s
// CHECK-MTD-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 4: /MD uses release library
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RELEASE %s
// CHECK-RELEASE: "-defaultlib:LLVMSYCL.lib"
// CHECK-RELEASE-NOT: LLVMSYCLd.lib

/// Test 5: /MDd uses debug library
// RUN: %clang_cl -### -fsycl /MDd --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG: "-defaultlib:LLVMSYCLd.lib"
// CHECK-DEBUG-NOT: "-defaultlib:LLVMSYCL.lib"

/// Test 6: -fms-runtime-lib=static is rejected
// RUN: not %clang_cl -fsycl -fms-runtime-lib=static \
// RUN:   --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-STATIC-ERROR %s
// CHECK-STATIC-ERROR: error: SYCL requires dynamic C++ runtime

/// Test 7: -fms-runtime-lib=dll_dbg uses debug library
// RUN: %clang_cl -### -fsycl -fms-runtime-lib=dll_dbg \
// RUN:   --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FLAG-DEBUG %s
// CHECK-FLAG-DEBUG: "-defaultlib:LLVMSYCLd.lib"

/// Test 8: LNK4078 warning is suppressed
// RUN: %clang_cl -### -fsycl --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-IGNORE %s
// CHECK-IGNORE: "/IGNORE:4078"

/// Test 9: -nolibsycl skips library linking and CRT check
// RUN: %clang_cl -### -fsycl -nolibsycl /MT \
// RUN:   --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOLIBSYCL %s
// CHECK-NOLIBSYCL-NOT: error:
// CHECK-NOLIBSYCL-NOT: "-defaultlib:LLVMSYCL

/// Test 10: Explicit /MD results in correct libraries
// RUN: %clang_cl -### -fsycl /MD --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EXPLICIT-MD %s
// CHECK-EXPLICIT-MD: clang-linker-wrapper"
// CHECK-EXPLICIT-MD-SAME: "-defaultlib:LLVMSYCL.lib"
// CHECK-EXPLICIT-MD-SAME: "-defaultlib:msvcrt"

/// Test 11: Library search path is added
// RUN: %clang_cl -### -fsycl --target=x86_64-pc-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LIBPATH %s
// CHECK-LIBPATH: "-libpath:{{.*}}{{[/\\]+}}lib"
