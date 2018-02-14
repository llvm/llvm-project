// RUN: mkdir -p %T/invalid-files
// RUN: cp %S/Inputs/invalid-files/invalid-files.yaml %T/invalid-files/invalid-files.yaml
// RUN: clang-apply-replacements %T/invalid-files
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls %T/invalid-files/invalid-files.yaml
