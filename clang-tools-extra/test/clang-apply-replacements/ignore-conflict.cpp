// RUN: mkdir -p %t.dir/Inputs/ignore-conflict
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/ignore-conflict/ignore-conflict.cpp > %t.dir/Inputs/ignore-conflict/ignore-conflict.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/ignore-conflict#" %S/Inputs/ignore-conflict/file1.yaml > %t.dir/Inputs/ignore-conflict/file1.yaml
// RUN: clang-apply-replacements --ignore-insert-conflict %t.dir/Inputs/ignore-conflict
// RUN: FileCheck -input-file=%t.dir/Inputs/ignore-conflict/ignore-conflict.cpp %S/Inputs/ignore-conflict/ignore-conflict.cpp
