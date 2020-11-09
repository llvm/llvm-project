// Tests for -fprofile-generate and -fprofile-use flag compatibility. These two
// flags behave similarly to their GCC counterparts:
//
// -fprofile-generate         Generates the profile file ./default.profraw
// -fprofile-generate=<dir>   Generates the profile file <dir>/default.profraw
// -fprofile-use              Uses the profile file ./default.profdata
// -fprofile-use=<dir>        Uses the profile file <dir>/default.profdata
// -fprofile-use=<dir>/file   Uses the profile file <dir>/file

// RUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-GEN %s
// RUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-GEN %s
// PROFILE-GEN: @__profc_main = {{(private|internal)}} global [2 x i64] zeroinitializer, section
// PROFILE-GEN: @__profd_main =

// Check that -fprofile-generate=/path/to generates /path/to/default.profraw
// RUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate=/path/to -fno-experimental-new-pass-manager | FileCheck -check-prefixes=PROFILE-GEN,PROFILE-GEN-EQ %s
// RxUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate=/path/to -fexperimental-new-pass-manager | FileCheck -check-prefixes=PROFILE-GEN,PROFILE-GEN-EQ %s
// PROFILE-GEN-EQ: constant [{{.*}} x i8] c"/path/to{{/|\\\\}}{{.*}}\00"

// Check that -fprofile-use=some/path reads some/path/default.profdata
// This uses Clang FE format profile.
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility.proftext -o %t.dir/some/path/default.profdata
// RUN: %clang %s -o - -Xclang -disable-llvm-passes -emit-llvm -S -fprofile-use=%t.dir/some/path -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE %s
// RUN: %clang %s -o - -Xclang -disable-llvm-passes -emit-llvm -S -fprofile-use=%t.dir/some/path -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE %s

// Check that -fprofile-use=some/path/file.prof reads some/path/file.prof
// This uses Clang FE format profile.
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility.proftext -o %t.dir/some/path/file.prof
// RUN: %clang %s -o - -Xclang -disable-llvm-passes -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE %s
// RUN: %clang %s -o - -Xclang -disable-llvm-passes -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE %s
// PROFILE-USE: = !{!"branch_weights", i32 101, i32 2}

// Check that -fprofile-use=some/path reads some/path/default.profdata
// This uses LLVM IR format profile.
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility_IR.proftext -o %t.dir/some/path/default.profdata
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s

// Check that -fprofile-use=some/path/file.prof reads some/path/file.prof
// This uses LLVM IR format profile.
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility_IR.proftext -o %t.dir/some/path/file.prof
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s
//
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility_IR_entry.proftext -o %t.dir/some/path/file.prof
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fno-experimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s
// RUN: %clang %s -o - -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof -fexperimental-new-pass-manager | FileCheck -check-prefix=PROFILE-USE-IR %s

// PROFILE-USE-IR: = !{!"branch_weights", i32 100, i32 1}

int X = 0;

int main() {
  int i;
  for (i = 0; i < 100; i++)
    X += i;
  return 0;
}
