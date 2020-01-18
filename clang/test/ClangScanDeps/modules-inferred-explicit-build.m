// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" -e "s|FRAMEWORKS|%/S/Inputs/frameworks|g" \
// RUN:   %S/Inputs/modules_inferred_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -full-command-line \
// RUN:   -mode preprocess-minimized-sources -format experimental-full > %t.db
// RUN: %S/module-deps-to-rsp.py %t.db --module-name=Inferred > %t.inferred.rsp
// RUN: %S/module-deps-to-rsp.py %t.db --tu-index=0 > %t.tu.rsp
// RUN: %clang_cc1 -x objective-c -E %t.dir/modules_cdb_input.cpp \
// RUN:   -F%S/Inputs/frameworks -fmodules -fimplicit-module-maps \
// RUN:   @%t.inferred.rsp
// RUN: %clang_cc1 -x objective-c -E %t.dir/modules_cdb_input.cpp \
// RUN:   -F%S/Inputs/frameworks -fmodules -fimplicit-module-maps @%t.tu.rsp

#include <Inferred/Inferred.h>

inferred a = 0;
