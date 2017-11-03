// RUN: clang-refactor extract -selection=test:%s %s -- -std=c++11 2>&1 | grep -v CHECK | FileCheck %s

struct Rectangle { int width, height; };

void extractStatement(const Rectangle &r) {
  /*range adeclstmt=->+0:59*/int area = r.width * r.height;
}
// CHECK: 1 'adeclstmt' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: int area = r.width * r.height;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatement(const Rectangle &r) {
// CHECK-NEXT:   /*range adeclstmt=->+0:59*/extracted();{{$}}
// CHECK-NEXT: }

void extractStatementNoSemiIf(const Rectangle &r) {
  /*range bextractif=->+2:4*/if (r.width) {
    int x = r.height;
  }
}
// CHECK: 1 'bextractif' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: if (r.width) {
// CHECK-NEXT: int x = r.height;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNoSemiIf(const Rectangle &r) {
// CHECK-NEXT:   /*range bextractif=->+2:4*/extracted();{{$}}
// CHECK-NEXT: }

void extractStatementDontExtraneousSemi(const Rectangle &r) {
  /*range cextractif=->+2:4*/if (r.width) {
    int x = r.height;
  } ;
} //^ This semicolon shouldn't be extracted.
// CHECK: 1 'cextractif' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: if (r.width) {
// CHECK-NEXT: int x = r.height;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementDontExtraneousSemi(const Rectangle &r) {
// CHECK-NEXT: extracted(); ;{{$}}
// CHECK-NEXT: }

void extractStatementNotSemiSwitch() {
  /*range dextract=->+5:4*/switch (2) {
  case 1:
    break;
  case 2:
    break;
  }
}
// CHECK: 1 'dextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: switch (2) {
// CHECK-NEXT: case 1:
// CHECK-NEXT:   break;
// CHECK-NEXT: case 2:
// CHECK-NEXT:   break;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNotSemiSwitch() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void extractStatementNotSemiWhile() {
  /*range eextract=->+2:4*/while (true) {
    int x = 0;
  }
}
// CHECK: 1 'eextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: while (true) {
// CHECK-NEXT: int x = 0;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNotSemiWhile() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void extractStatementNotSemiFor() {
  /*range fextract=->+1:4*/for (int i = 0; i < 10; ++i) {
  }
}
// CHECK: 1 'fextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNotSemiFor() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

struct XS {
  int *begin() { return 0; }
  int *end() { return 0; }
};

void extractStatementNotSemiRangedFor(XS xs) {
  /*range gextract=->+1:4*/for (int i : xs) {
  }
}
// CHECK: 1 'gextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: for (int i : xs) {
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNotSemiRangedFor(XS xs) {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void extractStatementNotSemiRangedTryCatch() {
  /*range hextract=->+3:4*/try { int x = 0; }
  catch (const int &i) {
    int y = i;
  }
}
// CHECK: 1 'hextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: try { int x = 0; }
// CHECK-NEXT: catch (const int &i) {
// CHECK-NEXT:   int y = i;
// CHECK-NEXT: }{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractStatementNotSemiRangedTryCatch() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void extractCantFindSemicolon() {
  /*range iextract=->+1:17*/do {
  } while (true)
  // Add a semicolon in both the extracted and original function as we don't
  // want to extract the semicolon below.
  ;
}
// CHECK: 1 'iextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: do {
// CHECK-NEXT: } while (true);{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractCantFindSemicolon() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: //
// CHECK-NEXT: //
// CHECK-NEXT: ;
// CHECK-NEXT: }

void extractFindSemicolon() {
  /*range jextract=->+1:17*/do {
  } while (true) /*grab*/ ;
}
// CHECK: 1 'jextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: do {
// CHECK-NEXT: } while (true) /*grab*/ ;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void extractFindSemicolon() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void call();

void careForNonCompoundSemicolons1() {
  /*range kextract=->+1:11*/if (true)
    call();
}
// CHECK: 1 'kextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: if (true)
// CHECK-NEXT: call();{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void careForNonCompoundSemicolons1() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: }

void careForNonCompoundSemicolons2() {
  /*range lextract=->+3:1*/for (int i = 0; i < 10; ++i)
    while (i != 0)
      ;
  // end right here111!
}
// CHECK: 1 'lextract' results:
// CHECK:      static void extracted() {
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: while (i != 0)
// CHECK-NEXT:   ;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: void careForNonCompoundSemicolons2() {
// CHECK-NEXT: extracted();{{$}}
// CHECK-NEXT: //
// CHECK-NEXT: }
