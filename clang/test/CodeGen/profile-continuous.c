// RUN: %clang %s -S -emit-llvm -fprofile-generate -fprofile-continuous -o - | FileCheck %s --check-prefix=IRPGO
// RUN: %clang %s -S -emit-llvm -fprofile-generate=mydir -fprofile-continuous -o - | FileCheck %s --check-prefix=IRPGO_EQ
// RUN: %clang %s -S -emit-llvm -fcs-profile-generate -fprofile-continuous -O -o - | FileCheck %s --check-prefix=CSIRPGO
// RUN: %clang %s -S -emit-llvm -fprofile-instr-generate -fprofile-continuous -o - | FileCheck %s --check-prefix=CLANG_PGO
// RUN: %clang %s -S -emit-llvm -fprofile-instr-generate= -fprofile-continuous -o - | FileCheck %s --check-prefix=CLANG_PGO
// RUN: %clang %s -S -emit-llvm -fprofile-instr-generate=foo.profraw -fprofile-continuous -o - | FileCheck %s --check-prefix=CLANG_PGO_EQ

// RUN: not %clang -### %s -fprofile-continuous -c 2>&1 | FileCheck %s --check-prefix=ERROR

// IRPGO: @__llvm_profile_filename = {{.*}} c"%cdefault_%m.profraw\00"
// IRPGO_EQ: @__llvm_profile_filename = {{.*}} c"%cmydir/default_%m.profraw\00"
// CSIRPGO: @__llvm_profile_filename = {{.*}} c"%cdefault_%m.profraw\00"
// CLANG_PGO: @__llvm_profile_filename = {{.*}} c"%cdefault.profraw\00"
// CLANG_PGO_EQ: @__llvm_profile_filename = {{.*}} c"%cfoo.profraw\00"
// ERROR: clang: error: invalid argument '-fprofile-continuous' only allowed with '-fprofile-generate, -fprofile-instr-generate, or -fcs-profile-generate'
void foo(){}
