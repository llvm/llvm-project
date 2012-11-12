// Note: the run lines follow all tests, since line/column matter here

id testCompleteAfterAtSign() {
  return @"";
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:4:11 %s -fno-const-strings -o - | FileCheck -check-prefix=AT %s
// CHECK-AT: COMPLETION: Pattern : [#NSString *#]"<#string#>"
// CHECK-AT: COMPLETION: Pattern : [#id#](<#expression#>)
// CHECK-AT: COMPLETION: Pattern : [#NSArray *#][<#objects, ...#>]
// CHECK-AT: COMPLETION: Pattern : [#char[]#]encode(<#type-name#>)
// CHECK-AT: COMPLETION: Pattern : [#Protocol *#]protocol(<#protocol-name#>)
// CHECK-AT: COMPLETION: Pattern : [#SEL#]selector(<#selector#>)
// CHECK-AT: COMPLETION: Pattern : [#NSDictionary *#]{<#key#>: <#object, ...#>}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:4:11 %s -fconst-strings -o - | FileCheck -check-prefix=CONST-STRINGS %s
// CHECK-CONST-STRINGS: COMPLETION: Pattern : [#const char[]#]encode(<#type-name#>)
