// RUN: pp-trace -ignore FileChanged %s -undef -target x86_64 -std=c++11 | FileCheck --strict-whitespace %s

#define MACRO 1
int i = MACRO;
#if defined(MACRO)
#endif
#undef MACRO
#if defined(MACRO)
#endif
#define FUNCMACRO(ARG1) ARG1
int j = FUNCMACRO(1);
#define X X_IMPL(a+y,b) X_IMPL2(c)
#define X_IMPL(p1,p2)
#define X_IMPL2(p1)
X

// CHECK: ---
// CHECK-NEXT: - Callback: MacroDefined
// CHECK:        MacroNameTok: __STDC__
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK:        MacroNameTok: __STDC_HOSTED__
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK:        MacroNameTok: __cplusplus
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK:        MacroNameTok: __STDC_UTF_16__
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK:        MacroNameTok: __STDC_UTF_32__
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK:      - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroExpands
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:4:9", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:4:9"]
// CHECK-NEXT:   Args: (null)
// CHECK-NEXT: - Callback: Defined
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:5:5", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:5:19"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:5:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:5:4", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:6:1"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:6:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:5:2"
// CHECK-NEXT: - Callback: MacroUndefined
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT: - Callback: Defined
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: []
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:5", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:19"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:4", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:9:1"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:9:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:8:1", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:9:2"]
// CHECK-NEXT: - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: FUNCMACRO
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroExpands
// CHECK-NEXT:   MacroNameTok: FUNCMACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:11:9", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:11:20"]
// CHECK-NEXT:   Args: [1]
// CHECK-NEXT: - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: X
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: X_IMPL
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: X_IMPL2
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: MacroExpands
// CHECK-NEXT:   MacroNameTok: X
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-macro.cpp:15:1", "{{.*}}{{[/\\]}}pp-trace-macro.cpp:15:1"]
// CHECK-NEXT:   Args: (null)
// CHECK-NEXT: - Callback: MacroExpands
// CHECK-NEXT:   MacroNameTok: X_IMPL
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: [(nonfile), (nonfile)]
// CHECK-NEXT:   Args: [a <plus> y, b]
// CHECK-NEXT: - Callback: MacroExpands
// CHECK-NEXT:   MacroNameTok: X_IMPL2
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT:   Range: [(nonfile), (nonfile)]
// CHECK-NEXT:   Args: [c]
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
