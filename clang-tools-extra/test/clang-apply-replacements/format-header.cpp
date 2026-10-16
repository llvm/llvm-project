// RUN: mkdir -p %t.dir/Inputs/format_header_yes
// RUN: mkdir -p %t.dir/Inputs/format_header_no
//
//
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format_header/yes.cpp > %t.dir/Inputs/format_header_yes/yes.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format_header/no.cpp > %t.dir/Inputs/format_header_no/no.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/format_header_yes#" %S/Inputs/format_header/yes.yaml > %t.dir/Inputs/format_header_yes/yes.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/format_header_no#" %S/Inputs/format_header/no.yaml > %t.dir/Inputs/format_header_no/no.yaml
// RUN: clang-apply-replacements -format -style="{BasedOnStyle: llvm, SortIncludes: CaseSensitive}" %t.dir/Inputs/format_header_yes
// RUN: clang-apply-replacements %t.dir/Inputs/format_header_no
// RUN: FileCheck --strict-whitespace -input-file=%t.dir/Inputs/format_header_yes/yes.cpp %S/Inputs/format_header/yes.cpp
// RUN: FileCheck --strict-whitespace -input-file=%t.dir/Inputs/format_header_no/no.cpp %S/Inputs/format_header/no.cpp
//
