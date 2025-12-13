// RUN: mkdir -p %t.dir/Inputs/identical
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/identical/identical.cpp > %t.dir/Inputs/identical/identical.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/identical#" %S/Inputs/identical/file1.yaml > %t.dir/Inputs/identical/file1.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/identical#" %S/Inputs/identical/file2.yaml > %t.dir/Inputs/identical/file2.yaml
// RUN: clang-apply-replacements %t.dir/Inputs/identical
// RUN: FileCheck -input-file=%t.dir/Inputs/identical/identical.cpp %S/Inputs/identical/identical.cpp
