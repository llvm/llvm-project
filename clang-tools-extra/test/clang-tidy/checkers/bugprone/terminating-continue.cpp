// RUN: %check_clang_tidy %s bugprone-terminating-continue %t

void f() {
  do {
    continue;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
    // CHECK-FIXES: break;
  } while(false);

  do {
    continue;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
    // CHECK-FIXES: break;
  } while(0);

  do {
    continue;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
    // CHECK-FIXES: break;
  } while(nullptr);

  do {
    continue;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
    // CHECK-FIXES: break;
  } while(__null);


  do {
    int x = 1;
    if (x > 0) continue;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
    // CHECK-FIXES: if (x > 0) break;
  } while (false);

  switch (2) {
  case 2:
    do {
      continue; // LoopInSwitch
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'continue' in loop with false condition is equivalent to 'break' [bugprone-terminating-continue]
      // CHECK-FIXES: break; // LoopInSwitch
    } while (0);
  }
}

void g() {
  do {
    do {
      continue;
      int x = 1;
    } while (1 == 1);
  } while (false);

  do {
    for (int i = 0; i < 1; ++i) {
      continue;
      int x = 1;
    }
  } while (false);

  do {
    while (true) {
      continue;
      int x = 1;
    }
  } while (false);

  int v[] = {1,2,3,34};
  do {
    for (int n : v) {
      if (n>2) continue;
    }
  } while (false);

  do {
    switch (2) {
    case 1:
    case 2:
      continue;
    }
  } while (false);
}
