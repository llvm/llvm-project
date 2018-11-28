// RUN: rm -rf %T/clang-tidy/export-relpath
// RUN: mkdir -p %T/clang-tidy/export-relpath/subdir
// RUN: cp %s %T/clang-tidy/export-relpath/subdir/source.cpp
// RUN: echo '[{ "directory": "%T/clang-tidy/export-relpath/subdir", "command": "clang++ source.cpp", "file": "%T/clang-tidy/export-relpath/subdir/source.cpp"}]' > %T/clang-tidy/export-relpath/subdir/compile_commands.json
//
// Check that running clang-tidy in './subdir' and storing results
// in './fixes.yaml' works as expected.
//
// RUN: cd %T/clang-tidy/export-relpath
// RUN: clang-tidy -p subdir subdir/source.cpp -checks='-*,google-explicit-constructor,llvm-namespace-comment' -export-fixes=./fixes.yaml
// RUN: FileCheck -input-file=%T/clang-tidy/export-relpath/fixes.yaml -check-prefix=CHECK-YAML %s

namespace i {
void f(); // So that the namespace isn't empty.
}
// CHECK-YAML: ReplacementText: ' // namespace i'

class A { A(int i); };
// CHECK-YAML: ReplacementText: 'explicit '
