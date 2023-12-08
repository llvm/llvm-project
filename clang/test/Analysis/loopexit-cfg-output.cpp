// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCFG -analyzer-config cfg-loopexit=true %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5

// CHECK:       [B1]
// CHECK-NEXT:   1: ForStmt (LoopExit)
// CHECK-NEXT:   2: return;
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B2.1]++
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B3]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B3.1]++
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B4]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 12
// CHECK-NEXT:   4: [B4.2] < [B4.3]
// CHECK-NEXT:   T: for (...; [B4.4]; ...)
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 B1

// CHECK:       [B5]
// CHECK-NEXT:   1: 0
// CHECK-NEXT:   2: int i = 0;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_forloop1() {
  for (int i = 0; i < 12; i++) {
    i++;
  }
  return;
}

// CHECK:       [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B1]
// CHECK-NEXT:   1: ForStmt (LoopExit)
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B3]
// CHECK-NEXT:   T: for (; ; )
// CHECK-NEXT:   Preds (2): B2 B4
// CHECK-NEXT:   Succs (2): B2 NULL

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_forloop2() {
  for (;;)
    ;
}

// CHECK:       [B5 (ENTRY)]
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B1]
// CHECK-NEXT:   1: WhileStmt (LoopExit)
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B3]
// CHECK-NEXT:   1: int i;
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B4]
// CHECK-NEXT:   1: true
// CHECK-NEXT:   T: while [B4.1]
// CHECK-NEXT:   Preds (2): B2 B5
// CHECK-NEXT:   Succs (2): B3 NULL

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_while1() {
  while (true) {
    int i;
  }
}

// CHECK:       [B5 (ENTRY)]
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B1]
// CHECK-NEXT:   1: WhileStmt (LoopExit)
// CHECK-NEXT:   2: 2
// CHECK-NEXT:   3: int k = 2;
// CHECK-NEXT:   4: return;
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B3]
// CHECK-NEXT:   1: l
// CHECK-NEXT:   2: [B3.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 42
// CHECK-NEXT:   4: [B3.2] < [B3.3]
// CHECK-NEXT:   T: while [B3.4]
// CHECK-NEXT:   Preds (2): B2 B4
// CHECK-NEXT:   Succs (2): B2 B1

// CHECK:       [B4]
// CHECK-NEXT:   1: int l;
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_while2() {
  int l;
  while (l < 42)
    ;
  int k = 2;
  return;
}

// CHECK:       [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B1]
// CHECK-NEXT:   1: WhileStmt (LoopExit)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B3]
// CHECK-NEXT:   1: false
// CHECK-NEXT:   T: while [B3.1]
// CHECK-NEXT:   Preds (2): B2 B4
// CHECK-NEXT:   Succs (2): NULL B1

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_while3() {
  while (false) {
    ;
  }
}

// CHECK:       [B4 (ENTRY)]
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B1]
// CHECK-NEXT:   1: DoStmt (LoopExit)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   1: false
// CHECK-NEXT:   T: do ... while [B2.1]
// CHECK-NEXT:   Preds (2): B3 B4
// CHECK-NEXT:   Succs (2): NULL B1

// CHECK:       [B3]
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_dowhile1() {
  do {
  } while (false);
}

// CHECK:       [B6 (ENTRY)]
// CHECK-NEXT:   Succs (1): B5

// CHECK:       [B1]
// CHECK-NEXT:   1: DoStmt (LoopExit)
// CHECK-NEXT:   2: j
// CHECK-NEXT:   3: [B1.2]--
// CHECK-NEXT:   4: return;
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: [B2.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 20
// CHECK-NEXT:   4: [B2.2] < [B2.3]
// CHECK-NEXT:   T: do ... while [B2.4]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (2): B4 B1

// CHECK:       [B3]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: 2
// CHECK-NEXT:   3: [B3.1] += [B3.2]
// CHECK-NEXT:   Preds (2): B4 B5
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B4]
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B5]
// CHECK-NEXT:   1: 2
// CHECK-NEXT:   2: int j = 2;
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B3

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_dowhile2() {
  int j = 2;
  do {
    j += 2;
  } while (j < 20);
  j--;
  return;
}

// CHECK:       [B10 (ENTRY)]
// CHECK-NEXT:   Succs (1): B9

// CHECK:       [B1]
// CHECK-NEXT:   1: WhileStmt (LoopExit)
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B8

// CHECK:       [B3]
// CHECK-NEXT:   1: ForStmt (LoopExit)
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B4]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: [B4.1]++
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (1): B6

// CHECK:       [B5]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B5.1]++
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B6]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: [B6.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 6
// CHECK-NEXT:   4: [B6.2] < [B6.3]
// CHECK-NEXT:   T: for (...; [B6.4]; ...)
// CHECK-NEXT:   Preds (2): B4 B7
// CHECK-NEXT:   Succs (2): B5 B3

// CHECK:       [B7]
// CHECK-NEXT:   1: 1
// CHECK-NEXT:   2: int j = 1;
// CHECK-NEXT:   Preds (1): B8
// CHECK-NEXT:   Succs (1): B6

// CHECK:       [B8]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B8.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 2
// CHECK-NEXT:   4: [B8.2] < [B8.3]
// CHECK-NEXT:   T: while [B8.4]
// CHECK-NEXT:   Preds (2): B2 B9
// CHECK-NEXT:   Succs (2): B7 B1

// CHECK:       [B9]
// CHECK-NEXT:   1: 40
// CHECK-NEXT:   2: -[B9.1]
// CHECK-NEXT:   3: int i = -40;
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (1): B8

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void nested_loops1() {
  int i = -40;
  while (i < 2) {
    for (int j = 1; j < 6; j++)
      i++;
  }
}

// CHECK:       [B9 (ENTRY)]
// CHECK-NEXT:   Succs (1): B8

// CHECK:       [B1]
// CHECK-NEXT:   1: ForStmt (LoopExit)
// CHECK-NEXT:   Preds (1): B7
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: [B2.1]++
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B7

// CHECK:       [B3]
// CHECK-NEXT:   1: DoStmt (LoopExit)
// CHECK-NEXT:   2: i
// CHECK-NEXT:   3: [B3.2]--
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B2

// CHECK:       [B4]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B4.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 2
// CHECK-NEXT:   4: [B4.2] < [B4.3]
// CHECK-NEXT:   T: do ... while [B4.4]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (2): B6 B3

// CHECK:       [B5]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B5.1]++
// CHECK-NEXT:   Preds (2): B6 B7
// CHECK-NEXT:   Succs (1): B4

// CHECK:       [B6]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B5

// CHECK:       [B7]
// CHECK-NEXT:   1: j
// CHECK-NEXT:   2: [B7.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 6
// CHECK-NEXT:   4: [B7.2] < [B7.3]
// CHECK-NEXT:   T: for (...; [B7.4]; ...)
// CHECK-NEXT:   Preds (2): B2 B8
// CHECK-NEXT:   Succs (2): B5 B1

// CHECK:       [B8]
// CHECK-NEXT:   1: 40
// CHECK-NEXT:   2: -[B8.1]
// CHECK-NEXT:   3: int i = -40;
// CHECK-NEXT:   4: 1
// CHECK-NEXT:   5: int j = 1;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B7

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void nested_loops2() {
  int i = -40;
  for (int j = 1; j < 6; j++) {
    do {
      i++;
    } while (i < 2);
    i--;
  }
}

// CHECK:       [B12 (ENTRY)]
// CHECK-NEXT:   Succs (1): B11

// CHECK:       [B1]
// CHECK-NEXT:   1: WhileStmt (LoopExit)
// CHECK-NEXT:   2: return;
// CHECK-NEXT:   Preds (2): B3 B5
// CHECK-NEXT:   Succs (1): B0

// CHECK:       [B2]
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B5

// CHECK:       [B3]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (1): B1

// CHECK:       [B4]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B4.1]++
// CHECK-NEXT:   3: i
// CHECK-NEXT:   4: [B4.3] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   5: 2
// CHECK-NEXT:   6: [B4.4] % [B4.5]
// CHECK-NEXT:   7: [B4.6] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:   T: if [B4.7]
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (2): B3 B2

// CHECK:       [B5]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B5.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 5
// CHECK-NEXT:   4: [B5.2] < [B5.3]
// CHECK-NEXT:   T: while [B5.4]
// CHECK-NEXT:   Preds (2): B2 B6
// CHECK-NEXT:   Succs (2): B4 B1

// CHECK:       [B6]
// CHECK-NEXT:   1: ForStmt (LoopExit)
// CHECK-NEXT:   2: 1
// CHECK-NEXT:   3: int i = 1;
// CHECK-NEXT:   Preds (2): B8 B10
// CHECK-NEXT:   Succs (1): B5

// CHECK:       [B7]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B7.1]++
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B10

// CHECK:       [B8]
// CHECK-NEXT:   T: break;
// CHECK-NEXT:   Preds (1): B9
// CHECK-NEXT:   Succs (1): B6

// CHECK:       [B9]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B9.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 4
// CHECK-NEXT:   4: [B9.2] == [B9.3]
// CHECK-NEXT:   T: if [B9.4]
// CHECK-NEXT:   Preds (1): B10
// CHECK-NEXT:   Succs (2): B8 B7

// CHECK:       [B10]
// CHECK-NEXT:   1: i
// CHECK-NEXT:   2: [B10.1] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   3: 6
// CHECK-NEXT:   4: [B10.2] < [B10.3]
// CHECK-NEXT:   T: for (...; [B10.4]; ...)
// CHECK-NEXT:   Preds (2): B7 B11
// CHECK-NEXT:   Succs (2): B9 B6

// CHECK:       [B11]
// CHECK-NEXT:   1: 2
// CHECK-NEXT:   2: int i = 2;
// CHECK-NEXT:   Preds (1): B12
// CHECK-NEXT:   Succs (1): B10

// CHECK:       [B0 (EXIT)]
// CHECK-NEXT:   Preds (1): B1
void check_break()
{
  for(int i = 2; i < 6; i++) {
    if(i == 4)
      break;
  }

  int i = 1;
  while(i<5){
    i++;
    if(i%2)
      break;
  }
  
  return;
}
