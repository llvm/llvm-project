
#include <DependsOnModule/DependsOnModule.h>
@import DependsOnModule;
int glob;

// RUN: rm -rf %t.cache
// RUN: c-index-test -test-annotate-tokens=%s:2:1:5:1 %s -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:      | FileCheck %s

// CHECK:      Punctuation: "#" [2:1 - 2:2] inclusion directive=[[INC_DIR:DependsOnModule[/\\]DependsOnModule\.h \(.*/Modules/Inputs/DependsOnModule\.framework[/\\]Headers[/\\]DependsOnModule.h\)]]
// CHECK-NEXT: Identifier: "include" [2:2 - 2:9] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Punctuation: "<" [2:10 - 2:11] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Identifier: "DependsOnModule" [2:11 - 2:26] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Punctuation: "/" [2:26 - 2:27] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Identifier: "DependsOnModule" [2:27 - 2:42] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Punctuation: "." [2:42 - 2:43] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Identifier: "h" [2:43 - 2:44] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Punctuation: ">" [2:44 - 2:45] inclusion directive=[[INC_DIR]]
// CHECK-NEXT: Punctuation: "@" [3:1 - 3:2] ModuleImport=DependsOnModule:3:1
// CHECK-NEXT: Keyword: "import" [3:2 - 3:8] ModuleImport=DependsOnModule:3:1
// CHECK-NEXT: Identifier: "DependsOnModule" [3:9 - 3:24] ModuleImport=DependsOnModule:3:1
// CHECK-NEXT: Punctuation: ";" [3:24 - 3:25]
// CHECK-NEXT: Keyword: "int" [4:1 - 4:4] VarDecl=glob:4:5
// CHECK-NEXT: Identifier: "glob" [4:5 - 4:9] VarDecl=glob:4:5
// CHECK-NEXT: Punctuation: ";" [4:9 - 4:10]

// RUN: c-index-test -test-annotate-tokens=%S/../Modules/Inputs/Module.framework/Headers/Sub.h:1:1:3:1 %s -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:      | FileCheck %s -check-prefix=CHECK-MOD

// CHECK-MOD:      Punctuation: "#" [1:1 - 1:2] inclusion directive=[[INC_DIR:Module[/\\]Sub2\.h \(.*/Modules/Inputs/Module\.framework[/\\]Headers[/\\]Sub2.h\)]]
// CHECK-MOD-NEXT: Identifier: "include" [1:2 - 1:9] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Punctuation: "<" [1:10 - 1:11] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Identifier: "Module" [1:11 - 1:17] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Punctuation: "/" [1:17 - 1:18] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Identifier: "Sub2" [1:18 - 1:22] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Punctuation: "." [1:22 - 1:23] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Identifier: "h" [1:23 - 1:24] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Punctuation: ">" [1:24 - 1:25] inclusion directive=[[INC_DIR]]
// CHECK-MOD-NEXT: Keyword: "int" [2:1 - 2:4] VarDecl=Module_Sub:2:6
// CHECK-MOD-NEXT: Punctuation: "*" [2:5 - 2:6] VarDecl=Module_Sub:2:6
// CHECK-MOD-NEXT: Identifier: "Module_Sub" [2:6 - 2:16] VarDecl=Module_Sub:2:6
// CHECK-MOD-NEXT: Punctuation: ";" [2:16 - 2:17]

// RUN: c-index-test -cursor-at=%s:3:11 %s -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:     | FileCheck %s -check-prefix=CHECK-CURSOR

// CHECK-CURSOR:      3:1 ModuleImport=DependsOnModule:3:1 (Definition) Extent=[3:1 - 3:24] Spelling=DependsOnModule ([3:9 - 3:24]) ModuleName=DependsOnModule ({{.*}}DependsOnModule-{{[^.]*}}.pcm) system=0 Headers(2):
// CHECK-CURSOR-NEXT: {{.*}}other.h
// CHECK-CURSOR-NEXT: {{.*}}DependsOnModule.h
