// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" -e "s|FRAMEWORKS|%/S/Inputs/frameworks|g" -e "s|-E|-x objective-c -E|g" \
// RUN:   %S/Inputs/modules_inferred_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -format experimental-full -mode preprocess-dependency-directives > %t.db
// RUN: %deps-to-rsp %t.db --module-name=Inferred > %t.inferred.cc1.rsp
// RUN: %deps-to-rsp %t.db --module-name=System > %t.system.cc1.rsp
// RUN: %deps-to-rsp %t.db --tu-index=0 > %t.tu.rsp
// RUN: %clang @%t.inferred.cc1.rsp -pedantic -Werror
// RUN: %clang @%t.system.cc1.rsp -pedantic -Werror
// RUN: %clang @%t.tu.rsp -pedantic -Werror

#include <Inferred/Inferred.h>
#include <System/System.h>

inferred a = bigger_than_int;
