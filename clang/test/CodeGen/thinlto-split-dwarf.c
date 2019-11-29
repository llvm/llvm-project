// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux-gnu \
// RUN:   -flto=thin -emit-llvm-bc \
// RUN:   -o %t.o %s

// RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
// RUN:   -o %t2.index \
// RUN:   -r=%t.o,main,px

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
// RUN:   -o %t.native.o -split-dwarf-file %t.file.dwo \
// RUN:   -split-dwarf-output %t.output.dwo -x ir %t.o

// RUN: llvm-dwarfdump %t.native.o | FileCheck --check-prefix=DWARFv4-O %s
// RUN: llvm-dwarfdump %t.output.dwo | FileCheck --check-prefix=DWARFv4-DWO %s

// DWARFv4-O: DW_AT_GNU_dwo_name ("{{.*}}.file.dwo")
// DWARFv4-O-NOT: DW_TAG_subprogram
// DWARFv4-DWO: DW_TAG_subprogram

// RUN: %clang_cc1 -debug-info-kind=limited -dwarf-version=5 -triple x86_64-unknown-linux-gnu \
// RUN:   -flto=thin -emit-llvm-bc \
// RUN:   -o %t.o %s

// RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
// RUN:   -o %t2.index \
// RUN:   -r=%t.o,main,px

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
// RUN:   -o %t.native.o -dwarf-version=5 -split-dwarf-file %t.file.dwo \
// RUN:   -split-dwarf-output %t.output.dwo -x ir %t.o

// RUN: llvm-dwarfdump %t.native.o | FileCheck --check-prefix=DWARFv5-O %s
// RUN: llvm-dwarfdump %t.output.dwo | FileCheck --check-prefix=DWARFv5-DWO %s

// DWARFv5-O: DW_AT_dwo_name ("{{.*}}.file.dwo")
// DWARFv5-O-NOT: DW_TAG_subprogram
// DWARFv5-DWO: DW_TAG_subprogram

int main() {}
