// REQUIRES: shell

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/resource_directory/* %t

// Deduce the resource directory from the compiler path.
//
// With `%clang-scan-deps --resource-dir-recipe modify-compiler-path`, the
// resource directory should be identical to `%clang -print-resource-dir`.
// (Assuming both binaries are built from the same Git checkout.)
// Here we get the expected path by running `%clang -print-resource-dir` and
// then verify `%clang-scan-deps` arrives at the same path by calling the
// `Driver::GetResourcesPath` function.
//
// RUN: EXPECTED_RESOURCE_DIR=`%clang -print-resource-dir`
// RUN: sed -e "s|CLANG|%clang|g" -e "s|DIR|%/t|g" \
// RUN:   %S/Inputs/resource_directory/cdb.json.template > %t/cdb_path.json
//
// RUN: clang-scan-deps -compilation-database %t/cdb_path.json --format experimental-full \
// RUN:   --resource-dir-recipe modify-compiler-path > %t/result_path.json
// RUN: cat %t/result_path.json | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s --check-prefix=CHECK-PATH -DEXPECTED_RESOURCE_DIR="$EXPECTED_RESOURCE_DIR"
// CHECK-PATH:      "-resource-dir"
// CHECK-PATH-NEXT: "[[EXPECTED_RESOURCE_DIR]]"

// Run the compiler and ask it for the resource directory.
//
// With `%clang-scan-deps --resource-dir-recipe invoke-compiler`, the resource
// directory should be identical to `<clang> -print-resource-dir`, where <clang>
// is an arbitrary version of Clang. (This configuration is not really supported.)
// Here we hard-code the expected path into `%t/compiler` and then verify
// `%clang-scan-deps` arrives at the path by actually running the executable.
//
// RUN: EXPECTED_RESOURCE_DIR="/custom/compiler/resources"
// RUN: echo "#!/bin/sh"                      > %t/compiler
// RUN: echo "echo '$EXPECTED_RESOURCE_DIR'" >> %t/compiler
// RUN: chmod +x %t/compiler
// RUN: sed -e "s|CLANG|%/t/compiler|g" -e "s|DIR|%/t|g" \
// RUN:   %S/Inputs/resource_directory/cdb.json.template > %t/cdb_invocation.json
//
// RUN: clang-scan-deps -compilation-database %t/cdb_invocation.json --format experimental-full \
// RUN:   --resource-dir-recipe invoke-compiler > %t/result_invocation.json
// RUN: cat %t/result_invocation.json | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s --check-prefix=CHECK-PATH -DEXPECTED_RESOURCE_DIR="$EXPECTED_RESOURCE_DIR"
// CHECK-INVOCATION:      "-resource-dir"
// CHECK-INVOCATION-NEXT: "[[EXPECTED_RESOURCE_DIR]]"
