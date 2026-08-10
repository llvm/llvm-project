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
int *ptr_return() {
  // line 37
}
struct Cls {};
int Cls::*memptr_return() {
  // line 41
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:28:1 %s -o - | FileCheck -check-prefix=RETURN-VAL %s
// RETURN-VAL-NOT: COMPLETION: Pattern : return;
// RETURN-VAL-NOT: COMPLETION: Pattern : return false;
// RETURN-VAL-NOT: COMPLETION: Pattern : return true;
// RETURN-VAL-NOT: COMPLETION: Pattern : return nullptr;
// RETURN-VAL: COMPLETION: Pattern : return <#expression#>;{{$}}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:31:1 %s -o - | FileCheck -check-prefix=RETURN-VOID %s
// RETURN-VOID-NOT: COMPLETION: Pattern : return false;
// RETURN-VOID-NOT: COMPLETION: Pattern : return true;
// RETURN-VOID-NOT: COMPLETION: Pattern : return <#expression#>;
// RETURN-VOID-NOT: COMPLETION: Pattern : return nullptr;
// RETURN-VOID: COMPLETION: Pattern : return;{{$}}

// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:34:1 %s -o - | FileCheck -check-prefix=RETURN-BOOL %s
// RETURN-BOOL-NOT: COMPLETION: Pattern : return;
// RETURN-BOOL-NOT: COMPLETION: Pattern : return nullptr;
// RETURN-BOOL: COMPLETION: Pattern : return <#expression#>;{{$}}
// RETURN-BOOL: COMPLETION: Pattern : return false;{{$}}
// RETURN-BOOL: COMPLETION: Pattern : return true;{{$}}

// Check both pointer and member pointer return types.
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -code-completion-patterns -code-completion-at=%s:37:1 %s -o - | FileCheck -check-prefix=RETURN-PTR %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -code-completion-patterns -code-completion-at=%s:41:1 %s -o - | FileCheck -check-prefix=RETURN-PTR %s
// RETURN-PTR-NOT: COMPLETION: Pattern : return false;{{$}}
// RETURN-PTR-NOT: COMPLETION: Pattern : return true;{{$}}
// RETURN-PTR-NOT: COMPLETION: Pattern : return;
// RETURN-PTR: COMPLETION: Pattern : return <#expression#>;{{$}}
// RETURN-PTR: COMPLETION: Pattern : return nullptr;

// 'return nullptr' is not available before C++11.
// RUN: %clang_cc1 -fsyntax-only -std=c++03 -code-completion-patterns -code-completion-at=%s:37:1 %s -o - | FileCheck -check-prefix=RETURN-PTR-STD03 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++03 -code-completion-patterns -code-completion-at=%s:41:1 %s -o - | FileCheck -check-prefix=RETURN-PTR-STD03 %s
// RETURN-PTR-STD03-NOT: COMPLETION: Pattern : return nullptr;

void something();

void unbraced_if() {
  if (true)
    something();
  // line 83
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:83:3 %s -o - | FileCheck -check-prefix=UNBRACED-IF %s
// UNBRACED-IF: COMPLETION: Pattern : else
// UNBRACED-IF-NEXT: <#statement#>;
// UNBRACED-IF: COMPLETION: Pattern : else if (<#condition#>)
// UNBRACED-IF-NEXT: <#statement#>;

void braced_if() {
  if (true) {
    something();
  }
  // line 95
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:95:3 %s -o - | FileCheck -check-prefix=BRACED-IF %s
// BRACED-IF: COMPLETION: Pattern : else {
// BRACED-IF-NEXT: <#statements#>
// BRACED-IF-NEXT: }
// BRACED-IF: COMPLETION: Pattern : else if (<#condition#>) {
// BRACED-IF-NEXT: <#statements#>
// BRACED-IF-NEXT: }
