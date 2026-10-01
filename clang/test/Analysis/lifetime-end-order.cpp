// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder %s 2>&1 \
// RUN:   -analyzer-config debug.AnalysisOrder:LifetimeEnd=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:EndFunction=true \
// RUN:   | FileCheck %s

// Tests when does the check::LifetimeEnd callback fires compared to
// other callbacks.

// Local integers end their lifetime at the closing brace of the
// inner block.
void testLocalInts() {
  {
    int i = 4, y = 5;
  }
}

// The loop variable's lifetime ends once at the point of the
// loop's exit.
void testLoopVar() {
  for(int i = 0; i < 3; ++i) {}
}

struct S {
  ~S();
};

// The lifetime of the object ends after its destructor returns.
void testDestrAndLifetimeEnd() {
  S s;
}

// Local integers lifetime ends at the return statement.
int testEarlyReturn() {
  {
    int i = 5;
    return i;
  }
}

// testEarlyReturn()
// CHECK:      LifetimeEnd
// CHECK-NEXT: EndFunction
// CHECK-NEXT: ReturnStmt: yes
// CHECK-NEXT: CFGElement: CFGLifetimeEnds

// testDestrAndLifetimeEnd()
// CHECK:      PreCall (S::S) [CXXConstructorCall]
// CHECK-NEXT: EndFunction
// CHECK-NEXT: ReturnStmt: no
// CHECK-NEXT: PostCall (S::S) [CXXConstructorCall]
// CHECK-NEXT: PreCall (S::~S) [CXXDestructorCall]
// CHECK-NEXT: PostCall (S::~S) [CXXDestructorCall]
// CHECK-NEXT: LifetimeEnd
// CHECK-NEXT: EndFunction
// CHECK-NEXT: ReturnStmt: no

// testLoopVar()
// CHECK:      LifetimeEnd
// CHECK-NEXT: EndFunction
// CHECK-NEXT: ReturnStmt: no

// testLocalInts()
// CHECK:      LifetimeEnd
// CHECK-NEXT: LifetimeEnd
// CHECK-NEXT: EndFunction
// CHECK-NEXT: ReturnStmt: no

// There should not be further LifetimeEnd elementes.
// CHECK-NOT: LifetimeEnd
