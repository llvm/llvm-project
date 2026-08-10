
#include "targeted-top.h"
#include "targeted-preamble.h"

int LocalVar1;
int LocalVar2;

// RUN: c-index-test -write-pch %t.h.pch %S/targeted-top.h
// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-annotate-tokens=%s:1:1:7:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -test-annotate-tokens=%s:1:1:7:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=LOCAL

// LOCAL: Punctuation: "#" [2:1 - 2:2] inclusion directive=targeted-top.h
// LOCAL: Identifier: "include" [2:2 - 2:9] inclusion directive=targeted-top.h
// LOCAL: Literal: ""targeted-top.h"" [2:10 - 2:26] inclusion directive=targeted-top.h
// LOCAL: Punctuation: "#" [3:1 - 3:2] inclusion directive=targeted-preamble.h
// LOCAL: Identifier: "include" [3:2 - 3:9] inclusion directive=targeted-preamble.h
// LOCAL: Literal: ""targeted-preamble.h"" [3:10 - 3:31] inclusion directive=targeted-preamble.h
// LOCAL: Keyword: "int" [5:1 - 5:4] VarDecl=LocalVar1:5:5
// LOCAL: Identifier: "LocalVar1" [5:5 - 5:14] VarDecl=LocalVar1:5:5
// LOCAL: Punctuation: ";" [5:14 - 5:15]
// LOCAL: Keyword: "int" [6:1 - 6:4] VarDecl=LocalVar2:6:5
// LOCAL: Identifier: "LocalVar2" [6:5 - 6:14] VarDecl=LocalVar2:6:5
// LOCAL: Punctuation: ";" [6:14 - 6:15]

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-annotate-tokens=%S/targeted-fields.h:1:1:4:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=FIELD

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -test-annotate-tokens=%S/targeted-fields.h:1:1:4:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=FIELD

// FIELD: Keyword: "int" [2:3 - 2:6] FieldDecl=z:2:7 (Definition)
// FIELD: Identifier: "z" [2:7 - 2:8] FieldDecl=z:2:7 (Definition)
// FIELD: Punctuation: ";" [2:8 - 2:9] StructDecl=Vector:13:9 (Definition)
// FIELD: Keyword: "int" [3:3 - 3:6] FieldDecl=w:3:7 (Definition)
// FIELD: Identifier: "w" [3:7 - 3:8] FieldDecl=w:3:7 (Definition)
// FIELD: Punctuation: ";" [3:8 - 3:9] StructDecl=Vector:13:9 (Definition)

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-annotate-tokens=%S/targeted-nested1.h:1:1:3:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -test-annotate-tokens=%S/targeted-nested1.h:1:1:3:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=TopVar  \
// RUN:  | FileCheck %s -check-prefix=NESTED

// NESTED: Keyword: "extern" [2:1 - 2:7]
// NESTED: Keyword: "int" [2:8 - 2:11] VarDecl=NestedVar1:2:12
// NESTED: Identifier: "NestedVar1" [2:12 - 2:22] VarDecl=NestedVar1:2:12
// NESTED: Punctuation: ";" [2:22 - 2:23]

// RUN: env CINDEXTEST_FAILONERROR=1 c-index-test -test-annotate-tokens=%S/targeted-top.h:1:1:12:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=vector_get_x  \
// RUN:  | FileCheck %s -check-prefix=TOP

// RUN: env CINDEXTEST_FAILONERROR=1 CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_NO_CACHING=1 \
// RUN:   c-index-test -test-annotate-tokens=%S/targeted-top.h:1:1:12:1 %s -include %t.h \
// RUN:    -Xclang -error-on-deserialized-decl=PreambleVar  \
// RUN:    -Xclang -error-on-deserialized-decl=NestedVar1  \
// RUN:    -Xclang -error-on-deserialized-decl=vector_get_x  \
// RUN:  | FileCheck %s -check-prefix=TOP

// TOP: Punctuation: "#" [2:1 - 2:2] preprocessing directive=
// TOP: Identifier: "ifndef" [2:2 - 2:8] preprocessing directive=
// TOP: Identifier: "TARGETED_TOP_H" [2:9 - 2:23] preprocessing directive=
// TOP: Punctuation: "#" [3:1 - 3:2] preprocessing directive=
// TOP: Identifier: "define" [3:2 - 3:8] preprocessing directive=
// TOP: Identifier: "TARGETED_TOP_H" [3:9 - 3:23] macro definition=TARGETED_TOP_H
// TOP: Punctuation: "#" [5:1 - 5:2] inclusion directive=targeted-nested1.h
// TOP: Identifier: "include" [5:2 - 5:9] inclusion directive=targeted-nested1.h
// TOP: Literal: ""targeted-nested1.h"" [5:10 - 5:30] inclusion directive=targeted-nested1.h
// TOP: Keyword: "enum" [7:1 - 7:5] EnumDecl=enum (unnamed at {{.*}}):7:1 (Definition)
// TOP: Punctuation: "{" [7:6 - 7:7] EnumDecl=enum (unnamed at {{.*}}):7:1 (Definition)
// TOP: Identifier: "VALUE" [8:3 - 8:8] EnumConstantDecl=VALUE:8:3 (Definition)
// TOP: Punctuation: "=" [8:9 - 8:10] EnumConstantDecl=VALUE:8:3 (Definition)
// TOP: Literal: "3" [8:11 - 8:12] IntegerLiteral=
// TOP: Punctuation: "}" [9:1 - 9:2] EnumDecl=enum (unnamed at {{.*}}):7:1 (Definition)
// TOP: Punctuation: ";" [9:2 - 9:3]
// TOP: Keyword: "extern" [11:1 - 11:7]
// TOP: Keyword: "int" [11:8 - 11:11] VarDecl=TopVar:11:12
// TOP: Identifier: "TopVar" [11:12 - 11:18] VarDecl=TopVar:11:12
// TOP: Punctuation: ";" [11:18 - 11:19]
