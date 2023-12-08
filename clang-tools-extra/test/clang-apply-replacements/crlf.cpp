// RUN: mkdir -p %T/Inputs/crlf
// RUN: cat %S/Inputs/crlf/crlf.cpp > %T/Inputs/crlf/crlf.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/crlf#" %S/Inputs/crlf/file1.yaml > %T/Inputs/crlf/file1.yaml
// RUN: clang-apply-replacements %T/Inputs/crlf
// RUN: diff %T/Inputs/crlf/crlf.cpp %S/Inputs/crlf/crlf.cpp.expected
