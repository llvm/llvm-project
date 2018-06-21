// RUN: mkdir -p %T/Inputs/identical
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/identical/identical.cpp > %T/Inputs/identical/identical.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/identical#" %S/Inputs/identical/file1.yaml > %T/Inputs/identical/file1.yaml
// RUN: clang-apply-replacements %T/Inputs/identical
// RUN: FileCheck -input-file=%T/Inputs/identical/identical.cpp %S/Inputs/identical/identical.cpp
