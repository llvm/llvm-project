// RUN: mkdir -p %t.dir/Inputs/basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic/basic.h > %t.dir/Inputs/basic/basic.h
// RUN: sed "s#\$(path)#%/t.dir/Inputs/basic#" %S/Inputs/basic/file1.yaml > %t.dir/Inputs/basic/file1.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/basic#" %S/Inputs/basic/file2.yaml > %t.dir/Inputs/basic/file2.yaml
// RUN: clang-apply-replacements %t.dir/Inputs/basic
// RUN: FileCheck -input-file=%t.dir/Inputs/basic/basic.h %S/Inputs/basic/basic.h
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls -1 %t.dir/Inputs/basic | FileCheck %s --check-prefix=YAML
//
// Check that the yaml files *are* deleted after running clang-apply-replacements with remove-change-desc-files.
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic/basic.h > %t.dir/Inputs/basic/basic.h
// RUN: clang-apply-replacements -remove-change-desc-files %t.dir/Inputs/basic
// RUN: ls -1 %t.dir/Inputs/basic | FileCheck %s --check-prefix=NO_YAML
//
// YAML: {{^file.\.yaml$}}
// NO_YAML-NOT: {{^file.\.yaml$}}
