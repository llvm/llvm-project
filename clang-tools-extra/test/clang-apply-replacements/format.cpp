// RUN: mkdir -p %t.dir/Inputs/format
//
// yes.cpp requires formatting after replacements are applied. no.cpp does not.
// The presence of no.cpp ensures that files that don't need formatting still
// have their new state written to disk after applying replacements.
//
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format/yes.cpp > %t.dir/Inputs/format/yes.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/format/no.cpp > %t.dir/Inputs/format/no.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/format#" %S/Inputs/format/yes.yaml > %t.dir/Inputs/format/yes.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/format#" %S/Inputs/format/no.yaml > %t.dir/Inputs/format/no.yaml
// RUN: clang-apply-replacements -format %t.dir/Inputs/format
// RUN: FileCheck --strict-whitespace -input-file=%t.dir/Inputs/format/yes.cpp %S/Inputs/format/yes.cpp
// RUN: FileCheck --strict-whitespace -input-file=%t.dir/Inputs/format/no.cpp %S/Inputs/format/no.cpp
//
// RUN not clang-apply-replacements -format=blah %t.dir/Inputs/format
