// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch/* %t

// Explicitly build the PCH:
//
// RUN: %clang -x c-header %t/pch.h -fmodules -gmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -o %t/pch.h.gch

// Scan dependencies of the TU:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_tu.json > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build
