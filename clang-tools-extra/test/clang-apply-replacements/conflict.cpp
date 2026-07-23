// RUN: mkdir -p %t.dir/Inputs/conflict
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file1.yaml > %t.dir/Inputs/conflict/file1.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file2.yaml > %t.dir/Inputs/conflict/file2.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/file3.yaml > %t.dir/Inputs/conflict/file3.yaml
// RUN: sed "s#\$(path)#%/S/Inputs/conflict#" %S/Inputs/conflict/expected.txt > %t.dir/Inputs/conflict/expected.txt
// RUN: not clang-apply-replacements %t.dir/Inputs/conflict > %t.dir/Inputs/conflict/output.txt 2>&1
// RUN: diff -b %t.dir/Inputs/conflict/output.txt %t.dir/Inputs/conflict/expected.txt
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files even when there is a failure.
// RUN: ls -1 %t.dir/Inputs/conflict | FileCheck %s --check-prefix=YAML
//
// Check that the yaml files *are* deleted after running clang-apply-replacements with remove-change-desc-files even when there is a failure.
// RUN: not clang-apply-replacements %t.dir/Inputs/conflict -remove-change-desc-files > %t.dir/Inputs/conflict/output.txt 2>&1
// RUN: ls -1 %t.dir/Inputs/conflict | FileCheck %s --check-prefix=NO_YAML
//
// YAML: {{^file.\.yaml$}}
// NO_YAML-NOT: {{^file.\.yaml$}}
