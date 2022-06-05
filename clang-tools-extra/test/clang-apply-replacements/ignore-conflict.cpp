// RUN: mkdir -p %T/Inputs/ignore-conflict
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/ignore-conflict/ignore-conflict.cpp > %T/Inputs/ignore-conflict/ignore-conflict.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/ignore-conflict#" %S/Inputs/ignore-conflict/file1.yaml > %T/Inputs/ignore-conflict/file1.yaml
// RUN: clang-apply-replacements --ignore-insert-conflict %T/Inputs/ignore-conflict
// RUN: FileCheck -input-file=%T/Inputs/ignore-conflict/ignore-conflict.cpp %S/Inputs/ignore-conflict/ignore-conflict.cpp
