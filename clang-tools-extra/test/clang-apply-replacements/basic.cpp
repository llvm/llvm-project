// RUN: mkdir -p %T/Inputs/basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic/basic.h > %T/Inputs/basic/basic.h
// RUN: sed "s#\$(path)#%/T/Inputs/basic#" %S/Inputs/basic/file1.yaml > %T/Inputs/basic/file1.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/basic#" %S/Inputs/basic/file2.yaml > %T/Inputs/basic/file2.yaml
// RUN: clang-apply-replacements %T/Inputs/basic
// RUN: FileCheck -input-file=%T/Inputs/basic/basic.h %S/Inputs/basic/basic.h
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls -1 %T/Inputs/basic | FileCheck %s --check-prefix=YAML
//
// Check that the yaml files *are* deleted after running clang-apply-replacements with remove-change-desc-files.
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic/basic.h > %T/Inputs/basic/basic.h
// RUN: clang-apply-replacements -remove-change-desc-files %T/Inputs/basic
// RUN: ls -1 %T/Inputs/basic | FileCheck %s --check-prefix=NO_YAML
//
// YAML: {{^file.\.yaml$}}
// NO_YAML-NOT: {{^file.\.yaml$}}
