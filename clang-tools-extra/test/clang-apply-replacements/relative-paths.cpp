// RUN: mkdir -p %t.dir/Inputs/relative-paths
// RUN: mkdir -p %t.dir/Inputs/relative-paths/subdir
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/relative-paths/basic.h > %t.dir/Inputs/relative-paths/basic.h
// RUN: sed "s#\$(path)#%/t.dir/Inputs/relative-paths#" %S/Inputs/relative-paths/file1.yaml > %t.dir/Inputs/relative-paths/file1.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/relative-paths#" %S/Inputs/relative-paths/file2.yaml > %t.dir/Inputs/relative-paths/file2.yaml
// RUN: clang-apply-replacements %t.dir/Inputs/relative-paths
// RUN: FileCheck -input-file=%t.dir/Inputs/relative-paths/basic.h %S/Inputs/relative-paths/basic.h
