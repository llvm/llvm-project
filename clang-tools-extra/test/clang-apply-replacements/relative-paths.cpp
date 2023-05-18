// RUN: mkdir -p %T/Inputs/relative-paths
// RUN: mkdir -p %T/Inputs/relative-paths/subdir
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/relative-paths/basic.h > %T/Inputs/relative-paths/basic.h
// RUN: sed "s#\$(path)#%/T/Inputs/relative-paths#" %S/Inputs/relative-paths/file1.yaml > %T/Inputs/relative-paths/file1.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/relative-paths#" %S/Inputs/relative-paths/file2.yaml > %T/Inputs/relative-paths/file2.yaml
// RUN: clang-apply-replacements %T/Inputs/relative-paths
// RUN: FileCheck -input-file=%T/Inputs/relative-paths/basic.h %S/Inputs/relative-paths/basic.h
