// RUN: %clang_cc1 -emit-llvm -fprofile-instrument=llvm -fprofile-continuous %s -o - | FileCheck %s --check-prefix=IRPGO
// RUN: %clang_cc1 -emit-llvm -fprofile-instrument=llvm -fprofile-continuous -fprofile-instrument-path=mydir/default_%m.profraw -mllvm -runtime-counter-relocation %s -o - \
// RUN:  | FileCheck %s --check-prefix=IRPGO_EQ
// RUN: %clang_cc1 -emit-llvm -O2 -fprofile-instrument=csllvm -fprofile-continuous %s -o - | FileCheck %s --check-prefix=CSIRPGO
// RUN: %clang_cc1 -emit-llvm -fprofile-instrument=clang -fprofile-continuous -fprofile-instrument-path=default.profraw %s -o - | FileCheck %s --check-prefix=CLANG_PGO

// IRPGO: @__llvm_profile_filename = {{.*}} c"%cdefault_%m.profraw\00"
// IRPGO_EQ: @__llvm_profile_filename = {{.*}} c"%cmydir/default_%m.profraw\00"
// CSIRPGO: @__llvm_profile_filename = {{.*}} c"%cdefault_%m.profraw\00"
// CLANG_PGO: @__llvm_profile_filename = {{.*}} c"%cdefault.profraw\00"
void foo(){}
