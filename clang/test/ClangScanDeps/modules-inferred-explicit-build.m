// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" -e "s|FRAMEWORKS|%/S/Inputs/frameworks|g" -e "s|-E|-x objective-c -E|g" \
// RUN:   %S/Inputs/modules_inferred_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -format experimental-full \
// RUN:   -mode preprocess-minimized-sources -generate-modules-path-args > %t.db
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t.db --module-name=Inferred > %t.inferred.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t.db --module-name=System > %t.system.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t.db --tu-index=0 > %t.tu.rsp
// RUN: %clang @%t.inferred.cc1.rsp -pedantic -Werror
// RUN: %clang @%t.system.cc1.rsp -pedantic -Werror
// RUN: %clang -x objective-c -fsyntax-only %t.dir/modules_cdb_input.cpp \
// RUN:   -F%S/Inputs/frameworks -fmodules -fimplicit-module-maps \
// RUN:   -pedantic -Werror @%t.tu.rsp

#include <Inferred/Inferred.h>
#include <System/System.h>

inferred a = bigger_than_int;
