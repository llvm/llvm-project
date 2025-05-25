// Test the output of -print-libgcc-file-name on Darwin.

//
// All platforms
//

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=x86_64-apple-macos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-MACOS %s
// CHECK-CLANGRT-MACOS: libclang_rt.osx.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-ios \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-IOS %s
// CHECK-CLANGRT-IOS: libclang_rt.ios.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-watchos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-WATCHOS %s
// CHECK-CLANGRT-WATCHOS: libclang_rt.watchos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-tvos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-TVOS %s
// CHECK-CLANGRT-TVOS: libclang_rt.tvos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-driverkit \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-DRIVERKIT %s
// CHECK-CLANGRT-DRIVERKIT: libclang_rt.driverkit.a

//
// Simulators
//

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-ios-simulator \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-IOS-SIMULATOR %s
// CHECK-CLANGRT-IOS-SIMULATOR: libclang_rt.iossim.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-watchos-simulator \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-WATCHOS-SIMULATOR %s
// CHECK-CLANGRT-WATCHOS-SIMULATOR: libclang_rt.watchossim.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-tvos-simulator \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-TVOS-SIMULATOR %s
// CHECK-CLANGRT-TVOS-SIMULATOR: libclang_rt.tvossim.a

// Check the sanitizer and profile variants
// While the driver also links in sanitizer-specific dylibs, the result of
// -print-libgcc-file-name is the path of the basic compiler-rt library.

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     -fsanitize=address --target=x86_64-apple-macos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-MACOS-SAN %s
// CHECK-CLANGRT-MACOS-SAN: libclang_rt.osx.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     -fsanitize=address --target=arm64-apple-ios \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-IOS-SAN %s
// CHECK-CLANGRT-IOS-SAN: libclang_rt.ios.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     -fsanitize=address --target=arm64-apple-watchos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-WATCHOS-SAN %s
// CHECK-CLANGRT-WATCHOS-SAN: libclang_rt.watchos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     -fsanitize=address --target=arm64-apple-tvos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-TVOS-SAN %s
// CHECK-CLANGRT-TVOS-SAN: libclang_rt.tvos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     -fsanitize=address --target=arm64-apple-driverkit \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-DRIVERKIT-SAN %s
// CHECK-CLANGRT-DRIVERKIT-SAN: libclang_rt.driverkit.a
