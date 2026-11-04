// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -gsrc-hash=md5 %s -o - | FileCheck %s

#include "Inputs/system-header-checksum.h"

int use_header(void) { return system_header_function(); }

// CHECK: !DIFile(filename: "{{.*}}system-header-checksum.h", directory: {{.*}}, checksumkind: CSK_MD5, checksum: "{{[0-9a-f]+}}")
