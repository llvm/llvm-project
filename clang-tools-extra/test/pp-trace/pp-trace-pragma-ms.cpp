// RUN: pp-trace -callbacks '*,-FileChanged,-MacroDefined' %s -- -target x86_64-unknown-windows-msvc -fms-extensions -w | FileCheck --strict-whitespace %s

#pragma comment(compiler, "compiler comment")
#pragma comment(exestr, "exestr comment")
#pragma comment(lib, "lib comment")
#pragma comment(linker, "linker comment")
#pragma comment(user, "user comment")

#pragma detect_mismatch("name argument", "value argument")

#pragma __debug(assert)

#pragma message("message argument")

#pragma warning(push, 1)
#pragma warning(pop)
#pragma warning(disable : 1 2 3 ; error : 4 5 6 ; suppress : 7 8 9)

// CHECK: ---
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:3:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaComment
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:3:9"
// CHECK-NEXT:   Kind: compiler
// CHECK-NEXT:   Str: compiler comment
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:4:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaComment
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:4:9"
// CHECK-NEXT:   Kind: exestr
// CHECK-NEXT:   Str: exestr comment
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:5:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaComment
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:5:9"
// CHECK-NEXT:   Kind: lib
// CHECK-NEXT:   Str: lib comment
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:6:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaComment
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:6:9"
// CHECK-NEXT:   Kind: linker
// CHECK-NEXT:   Str: linker comment
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:7:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaComment
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:7:9"
// CHECK-NEXT:   Kind: user
// CHECK-NEXT:   Str: user comment
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:9:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaDetectMismatch
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:9:9"
// CHECK-NEXT:   Name: name argument
// CHECK-NEXT:   Value: value argument
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:11:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:13:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaMessage
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:13:9"
// CHECK-NEXT:   Namespace: 
// CHECK-NEXT:   Kind: PMK_Message
// CHECK-NEXT:   Str: message argument
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:15:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaWarningPush
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:15:9"
// CHECK-NEXT:   Level: 1
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:16:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaWarningPop
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:16:9"
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:17:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaWarning
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:17:9"
// CHECK-NEXT:   WarningSpec: disable
// CHECK-NEXT:   Ids: [1, 2, 3]
// CHECK-NEXT: - Callback: PragmaWarning
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:17:9"
// CHECK-NEXT:   WarningSpec: error
// CHECK-NEXT:   Ids: [4, 5, 6]
// CHECK-NEXT: - Callback: PragmaWarning
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-ms.cpp:17:9"
// CHECK-NEXT:   WarningSpec: suppress
// CHECK-NEXT:   Ids: [7, 8, 9]
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
