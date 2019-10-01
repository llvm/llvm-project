void loops() {
  while (true) {
    // line 3
  }
  for (;;) {
    // line 6
  }
  do {
    // line 9
  } while (true);
  // line 11
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:3:1 %s -o - | FileCheck -check-prefix=LOOP %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:6:1 %s -o - | FileCheck -check-prefix=LOOP %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:9:1 %s -o - | FileCheck -check-prefix=LOOP %s
// LOOP: COMPLETION: Pattern : break;{{$}}
// LOOP: COMPLETION: Pattern : continue;{{$}}
// LOOP: COMPLETION: Pattern : goto <#label#>;{{$}}
// LOOP: COMPLETION: Pattern : return;{{$}}
//
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:11:1 %s -o - | FileCheck -check-prefix=OUTSIDE-LOOP %s
// OUTSIDE-LOOP-NOT: COMPLETION: Pattern : break;{{$}}
// OUTSIDE-LOOP-NOT: COMPLETION: Pattern : continue;{{$}}
// OUTSIDE-LOOP: COMPLETION: Pattern : goto <#label#>;{{$}}
// OUTSIDE-LOOP: COMPLETION: Pattern : return;{{$}}

int value_return() {
  // line 28
}
void void_return() {
  // line 31
}
bool bool_return() {
  // line 34
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:28:1 %s -o - | FileCheck -check-prefix=RETURN-VAL %s
// RETURN-VAL-NOT: COMPLETION: Pattern : return;
// RETURN-VAL-NOT: COMPLETION: Pattern : return false;
// RETURN-VAL-NOT: COMPLETION: Pattern : return true;
// RETURN-VAL: COMPLETION: Pattern : return <#expression#>;{{$}}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:31:1 %s -o - | FileCheck -check-prefix=RETURN-VOID %s
// RETURN-VOID-NOT: COMPLETION: Pattern : return false;
// RETURN-VOID-NOT: COMPLETION: Pattern : return true;
// RETURN-VOID-NOT: COMPLETION: Pattern : return <#expression#>;
// RETURN-VOID: COMPLETION: Pattern : return;{{$}}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:34:1 %s -o - | FileCheck -check-prefix=RETURN-BOOL %s
// RETURN-BOOL-NOT: COMPLETION: Pattern : return;
// RETURN-BOOL: COMPLETION: Pattern : return <#expression#>;{{$}}
// RETURN-BOOL: COMPLETION: Pattern : return false;{{$}}
// RETURN-BOOL: COMPLETION: Pattern : return true;{{$}}
