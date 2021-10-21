// REQUIRES: shell

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/resource_directory/* %t

// Deduce the resource directory from the compiler path.
//
// RUN: sed -e "s|CLANG|/our/custom/bin/clang|g" -e "s|DIR|%/t|g" \
// RUN:   %S/Inputs/resource_directory/cdb.json.template > %t/cdb_path.json
// RUN: clang-scan-deps -compilation-database %t/cdb_path.json --format experimental-full \
// RUN:   --resource-dir-recipe modify-compiler-path > %t/result_path.json
// RUN: cat %t/result_path.json | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-PATH
// CHECK-PATH:      "-resource-dir"
// CHECK-PATH-NEXT: "/our/custom/lib{{.*}}"

// Run the compiler and ask it for the resource directory.
//
// RUN: chmod +x %t/compiler
// RUN: sed -e "s|CLANG|%/t/compiler|g" -e "s|DIR|%/t|g" \
// RUN:   %S/Inputs/resource_directory/cdb.json.template > %t/cdb_invocation.json
// RUN: clang-scan-deps -compilation-database %t/cdb_invocation.json --format experimental-full \
// RUN:   --resource-dir-recipe invoke-compiler > %t/result_invocation.json
// RUN: cat %t/result_invocation.json | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-INVOCATION
// CHECK-INVOCATION:      "-resource-dir"
// CHECK-INVOCATION-NEXT: "/custom/compiler/resources"
