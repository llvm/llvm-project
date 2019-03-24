// RUN: pp-trace -callbacks '*,-FileChanged' %s -- -undef -target x86_64 -std=c++11 | FileCheck --strict-whitespace %s

#if 1
#endif

#if 0
#endif

#if 1
#else
#endif

#if 0
#else
#endif

#if 1
#elif 1
#endif
#if 1
#elif 0
#endif

#if 0
#elif 1
#endif
#if 0
#elif 0
#endif
#if 1
#elif 1
#endif
#if 1
#elif 0
#endif

#if 0
#elif 1
#else
#endif
#if 0
#elif 0
#else
#endif
#if 1
#elif 1
#else
#endif
#if 1
#elif 0
#else
#endif

#define MACRO 1
#ifdef MACRO
#endif
#ifdef NO_MACRO
#endif
#ifndef MACRO
#endif
#ifndef NO_MACRO
#endif

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
// CHECK:      - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:3:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:3:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:3:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:4:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:3:2"
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:6:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:6:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:6:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:7:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:6:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:6:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:7:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:9:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:9:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:9:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Else
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:10:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:9:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:11:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:9:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:10:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:11:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Else
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:14:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:14:2"]
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:15:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:13:2"
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:17:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:17:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:17:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:18:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:18:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:18:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:17:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:19:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:17:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:18:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:19:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:20:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:20:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:20:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:21:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:21:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:21:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:20:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:22:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:20:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:21:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:22:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:25:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:25:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:25:7"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:25:2"]
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:26:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:24:2"
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:28:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:28:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:28:7"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:29:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:27:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:29:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:30:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:30:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:30:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:31:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:31:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:31:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:30:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:32:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:30:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:31:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:32:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:33:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:33:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:33:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:34:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:34:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:34:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:33:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:35:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:33:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:34:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:35:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:38:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:38:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:38:7"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:38:2"]
// CHECK-NEXT: - Callback: Else
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:39:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:40:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:37:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:39:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:40:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:5"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:42:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:42:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:42:7"]
// CHECK-NEXT:   ConditionValue: CVK_False
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:2"
// CHECK-NEXT: - Callback: Else
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:43:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:43:2"]
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:44:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:41:2"
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:45:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:45:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:45:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:46:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:46:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:46:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:45:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:48:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:45:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:46:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:48:2"]
// CHECK-NEXT: - Callback: If
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:49:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:49:5", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:49:5"]
// CHECK-NEXT:   ConditionValue: CVK_True
// CHECK-NEXT: - Callback: Elif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:50:2"
// CHECK-NEXT:   ConditionRange: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:50:7", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:50:8"]
// CHECK-NEXT:   ConditionValue: CVK_NotEvaluated
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:49:2"
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:52:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:49:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:50:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:52:2"]
// CHECK-NEXT: - Callback: MacroDefined
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDirective: MD_Define
// CHECK-NEXT: - Callback: Ifdef
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:55:2"
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:56:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:55:2"
// CHECK-NEXT: - Callback: Ifdef
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:57:2"
// CHECK-NEXT:   MacroNameTok: NO_MACRO
// CHECK-NEXT:   MacroDefinition: []
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:58:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:57:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:57:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:58:2"]
// CHECK-NEXT: - Callback: Ifndef
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:59:2"
// CHECK-NEXT:   MacroNameTok: MACRO
// CHECK-NEXT:   MacroDefinition: [(local)]
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:60:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:59:2"
// CHECK-NEXT: - Callback: SourceRangeSkipped
// CHECK-NEXT:   Range: ["{{.*}}{{[/\\]}}pp-trace-conditional.cpp:59:1", "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:60:2"]
// CHECK-NEXT: - Callback: Ifndef
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:61:2"
// CHECK-NEXT:   MacroNameTok: NO_MACRO
// CHECK-NEXT:   MacroDefinition: []
// CHECK-NEXT: - Callback: Endif
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:62:2"
// CHECK-NEXT:   IfLoc: "{{.*}}{{[/\\]}}pp-trace-conditional.cpp:61:2"
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
