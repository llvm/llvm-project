// Tests for -fprofile-generate and -fprofile-use flag compatibility. These two
// flags behave similarly to their GCC counterparts:
//
// -fprofile-generate         Generates the profile file ./default.profraw
// -fprofile-generate=<dir>   Generates the profile file <dir>/default.profraw
// -fprofile-use              Uses the profile file ./default.profdata
// -fprofile-use=<dir>        Uses the profile file <dir>/default.profdata
// -fprofile-use=<dir>/file   Uses the profile file <dir>/file

// FIXME: IRPGO shouldn't use the override API when no profraw name is given.
// Check that -fprofile-generate overrides the default profraw.
// RUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate | FileCheck -check-prefix=PROFILE-GEN %s
// PROFILE-GEN: call void @__llvm_profile_override_default_filename
// PROFILE-GEN: declare void @__llvm_profile_override_default_filename(i8*)

// Check that -fprofile-generate=/path/to generates /path/to/default.profraw
// RUN: %clang %s -c -S -o - -emit-llvm -fprofile-generate=/path/to | FileCheck -check-prefix=PROFILE-GEN-EQ %s
// PROFILE-GEN-EQ: private constant [25 x i8] c"/path/to{{/|\\5C}}default.profraw\00"
// PROFILE-GEN-EQ: call void @__llvm_profile_override_default_filename(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @0, i32 0, i32 0))
// PROFILE-GEN-EQ: declare void @__llvm_profile_override_default_filename(i8*)

// Check that -fprofile-use=some/path reads some/path/default.profdata
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility.proftext -o %t.dir/some/path/default.profdata
// RUN: %clang %s -o - -mllvm -disable-llvm-optzns -emit-llvm -S -fprofile-use=%t.dir/some/path | FileCheck -check-prefix=PROFILE-USE-2 %s
// PROFILE-USE-2: = !{!"branch_weights", i32 101, i32 2}

// Check that -fprofile-use=some/path/file.prof reads some/path/file.prof
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/some/path
// RUN: llvm-profdata merge %S/Inputs/gcc-flag-compatibility.proftext -o %t.dir/some/path/file.prof
// RUN: %clang %s -o - -mllvm -disable-llvm-optzns -emit-llvm -S -fprofile-use=%t.dir/some/path/file.prof | FileCheck -check-prefix=PROFILE-USE-3 %s
// PROFILE-USE-3: = !{!"branch_weights", i32 101, i32 2}

int X = 0;

int main() {
  int i;
  for (i = 0; i < 100; i++)
    X += i;
  return 0;
}
