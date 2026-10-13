// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyze-function="testLocalInts()"\
// RUN:   -analyzer-config debug.AnalysisOrder:LifetimeEnd=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:EndFunction=true\
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOCAL-INTS
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyze-function="testLoopVar()"\
// RUN:   -analyzer-config debug.AnalysisOrder:LifetimeEnd=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:EndFunction=true\
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOOP-VAR
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyze-function="testDestrAndLifetimeEnd()"\
// RUN:   -analyzer-config debug.AnalysisOrder:LifetimeEnd=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:EndFunction=true\
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHECK-DESTR
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyze-function="testEarlyReturn()"\
// RUN:   -analyzer-config debug.AnalysisOrder:LifetimeEnd=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:EndFunction=true\
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHECK-EARLY-RETURN

// Tests when does the check::LifetimeEnd callback fires compared to
// other callbacks.

// Local integers end their lifetime at the closing brace of the
// inner block.
void testLocalInts() {
  {
    int i = 4, y = 5;
  }
}

// CHECK-LOCAL-INTS:      LifetimeEnd
// CHECK-LOCAL-INTS-NEXT: LifetimeEnd
// CHECK-LOCAL-INTS-NEXT: EndFunction
// CHECK-LOCAL-INTS-NEXT: ReturnStmt: no
// CHECK-LOCAL-INTS-NOT:  LifetimeEnd

// The loop variable's lifetime ends once at the point of the
// loop's exit.
void testLoopVar() {
  for(int i = 0; i < 3; ++i) {}
}

// CHECK-LOOP-VAR:      LifetimeEnd
// CHECK-LOOP-VAR-NEXT: EndFunction
// CHECK-LOOP-VAR-NEXT: ReturnStmt: no
// CHECK-LOOP-VAR-NOT:  LifetimeEnd

struct S {
  ~S();
};

// The lifetime of the object ends after its destructor returns.
void testDestrAndLifetimeEnd() {
  S s;
}

// CHECK-DESTR:      PreCall (S::S) [CXXConstructorCall]
// CHECK-DESTR-NEXT: EndFunction
// CHECK-DESTR-NEXT: ReturnStmt: no
// CHECK-DESTR-NEXT: PostCall (S::S) [CXXConstructorCall]
// CHECK-DESTR-NEXT: PreCall (S::~S) [CXXDestructorCall]
// CHECK-DESTR-NEXT: PostCall (S::~S) [CXXDestructorCall]
// CHECK-DESTR-NEXT: LifetimeEnd
// CHECK-DESTR-NEXT: EndFunction
// CHECK-DESTR-NEXT: ReturnStmt: no
// CHECK-DESTR-NOT:  LifetimeEnd

// Local integers lifetime ends at the return statement.
int testEarlyReturn() {
  {
    int i = 5;
    return i;
  }
}

// CHECK-EARLY-RETURN:      LifetimeEnd
// CHECK-EARLY-RETURN-NEXT: EndFunction
// CHECK-EARLY-RETURN-NEXT: ReturnStmt: yes
// CHECK-EARLY-RETURN-NEXT: CFGElement: CFGLifetimeEnds
// CHECK-EARLY-RETURN-NOT:  LifetimeEnd
