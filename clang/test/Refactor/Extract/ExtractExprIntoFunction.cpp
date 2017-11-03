// RUN: clang-refactor extract -selection=test:%s %s -- -std=c++11 2>&1 | grep -v CHECK | FileCheck %s


void simpleExtractNoCaptures() {
  int i = /*range=->+0:33*/1 + 2;
}

// CHECK: 1 '' results:
// CHECK:      static int extracted() {
// CHECK-NEXT: return 1 + 2;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void simpleExtractNoCaptures() {
// CHECK-NEXT:   int i = /*range=->+0:33*/extracted();{{$}}
// CHECK-NEXT: }

void simpleExtractStmtNoCaptures() {
  /*range astatement=->+1:13*/int a = 1;
  int b = 2;
}
// CHECK: 1 'astatement' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: int a = 1;
// CHECK-NEXT: int b = 2;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void simpleExtractStmtNoCaptures() {
// CHECK-NEXT:   /*range astatement=->+1:13*/extracted();{{$}}
// CHECK-NEXT: }


void blankRangeNoExtraction() {
  int i = /*range blank=*/1 + 2;
}

// CHECK: 1 'blank' results:
// CHECK-NEXT: the provided selection does not overlap with the AST nodes of interest

int outOfBodyCodeNoExtraction = /*range out_of_body_expr=->+0:72*/1 + 2;

struct OutOfBodyStuff {
  int FieldInit = /*range out_of_body_expr=->+0:58*/1 + 2;

  void foo(int x =/*range out_of_body_expr=->+0:58*/1 + 2);
};

// CHECK: 3 'out_of_body_expr' results:
// CHECK: the selected code is not a part of a function's / method's body

void simpleExpressionNoExtraction() {
  int i = /*range simple_expr=->+0:41*/1 + /*range simple_expr=->+0:76*/(2);
  (void) /*range simple_expr=->+0:40*/i;
  (void)/*range simple_expr=->+0:47*/"literal";
  (void)/*range simple_expr=->+0:41*/'c';
}

// CHECK: 5 'simple_expr' results:
// CHECK-NEXT: the selected expression is too simple to extract
