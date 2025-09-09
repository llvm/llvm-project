// RUN: mkdir -p %t.dir/Inputs/crlf
// RUN: cp %S/Inputs/crlf/crlf.cpp %t.dir/Inputs/crlf/crlf.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/crlf#" %S/Inputs/crlf/file1.yaml > %t.dir/Inputs/crlf/file1.yaml
// RUN: clang-apply-replacements %t.dir/Inputs/crlf
// RUN: diff %t.dir/Inputs/crlf/crlf.cpp %S/Inputs/crlf/crlf.cpp.expected
