// RUN: mkdir -p %t.dir/invalid-files
// RUN: cp %S/Inputs/invalid-files/invalid-files.yaml %t.dir/invalid-files/invalid-files.yaml
// RUN: clang-apply-replacements %t.dir/invalid-files
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls %t.dir/invalid-files/invalid-files.yaml
