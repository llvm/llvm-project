// RUN: mkdir -p %T/Inputs/format_header_yes
// RUN: mkdir -p %T/Inputs/format_header_no
//
//
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format_header/yes.cpp > %T/Inputs/format_header_yes/yes.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format_header/no.cpp > %T/Inputs/format_header_no/no.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/format_header_yes#" %S/Inputs/format_header/yes.yaml > %T/Inputs/format_header_yes/yes.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/format_header_no#" %S/Inputs/format_header/no.yaml > %T/Inputs/format_header_no/no.yaml
// RUN: clang-apply-replacements -format -style="{BasedOnStyle: llvm, SortIncludes: CaseSensitive}" %T/Inputs/format_header_yes
// RUN: clang-apply-replacements %T/Inputs/format_header_no
// RUN: FileCheck --strict-whitespace -input-file=%T/Inputs/format_header_yes/yes.cpp %S/Inputs/format_header/yes.cpp
// RUN: FileCheck --strict-whitespace -input-file=%T/Inputs/format_header_no/no.cpp %S/Inputs/format_header/no.cpp
//
