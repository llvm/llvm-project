// RUN: mkdir -p %t.dir/Inputs/yml-basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/yml-basic/basic.h > %t.dir/Inputs/yml-basic/basic.h
// RUN: sed "s#\$(path)#%/t.dir/Inputs/yml-basic#" %S/Inputs/yml-basic/file1.yml > %t.dir/Inputs/yml-basic/file1.yml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/yml-basic#" %S/Inputs/yml-basic/file2.yml > %t.dir/Inputs/yml-basic/file2.yml
// RUN: clang-apply-replacements %t.dir/Inputs/yml-basic
// RUN: FileCheck -input-file=%t.dir/Inputs/yml-basic/basic.h %S/Inputs/yml-basic/basic.h
//
// Check that the yml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls -1 %t.dir/Inputs/yml-basic | FileCheck %s --check-prefix=YML
//
// Check that the yml files *are* deleted after running clang-apply-replacements with remove-change-desc-files.
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/yml-basic/basic.h > %t.dir/Inputs/yml-basic/basic.h
// RUN: clang-apply-replacements -remove-change-desc-files %t.dir/Inputs/yml-basic
// RUN: ls -1 %t.dir/Inputs/yml-basic | FileCheck %s --check-prefix=NO_YML
//
// YML: {{^file.\.yml$}}
// NO_YML-NOT: {{^file.\.yml$}}
