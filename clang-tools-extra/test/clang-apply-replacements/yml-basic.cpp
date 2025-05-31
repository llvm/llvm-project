// RUN: mkdir -p %T/Inputs/yml-basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/yml-basic/basic.h > %T/Inputs/yml-basic/basic.h
// RUN: sed "s#\$(path)#%/T/Inputs/yml-basic#" %S/Inputs/yml-basic/file1.yml > %T/Inputs/yml-basic/file1.yml
// RUN: sed "s#\$(path)#%/T/Inputs/yml-basic#" %S/Inputs/yml-basic/file2.yml > %T/Inputs/yml-basic/file2.yml
// RUN: clang-apply-replacements %T/Inputs/yml-basic
// RUN: FileCheck -input-file=%T/Inputs/yml-basic/basic.h %S/Inputs/yml-basic/basic.h
//
// Check that the yml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls -1 %T/Inputs/yml-basic | FileCheck %s --check-prefix=YML
//
// Check that the yml files *are* deleted after running clang-apply-replacements with remove-change-desc-files.
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/yml-basic/basic.h > %T/Inputs/yml-basic/basic.h
// RUN: clang-apply-replacements -remove-change-desc-files %T/Inputs/yml-basic
// RUN: ls -1 %T/Inputs/yml-basic | FileCheck %s --check-prefix=NO_YML
//
// YML: {{^file.\.yml$}}
// NO_YML-NOT: {{^file.\.yml$}}
