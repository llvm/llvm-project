// RUN: %check_clang_tidy %s cppcoreguidelines-avoid-goto %t

void noop() {}

int main() {
  noop();
  goto jump_to_me;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: avoid using 'goto' for flow control
  // CHECK-MESSAGES: [[@LINE+3]]:1: note: label defined here
  noop();

jump_to_me:;

jump_backwards:;
  noop();
  goto jump_backwards;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: avoid using 'goto' for flow control
  // CHECK-MESSAGES: [[@LINE-4]]:1: note: label defined here

  goto jump_in_line;
  ;
jump_in_line:;
  // CHECK-MESSAGES: [[@LINE-3]]:3: warning: avoid using 'goto' for flow control
  // CHECK-MESSAGES: [[@LINE-2]]:1: note: label defined here

  // Test the GNU extension https://gcc.gnu.org/onlinedocs/gcc/Labels-as-Values.html
some_label:;
  void *dynamic_label = &&some_label;

  // FIXME: `IndirectGotoStmt` is not detected.
  goto *dynamic_label;
}

void forward_jump_out_nested_loop() {
  int array[] = {1, 2, 3, 4, 5};
  for (int i = 0; i < 10; ++i) {
    noop();
    for (int j = 0; j < 10; ++j) {
      noop();
      if (i + j > 10)
        goto early_exit1;
    }
    noop();
  }

  for (int i = 0; i < 10; ++i) {
    noop();
    while (true) {
      noop();
      if (i > 5)
        goto early_exit1;
    }
    noop();
  }

  for (auto value : array) {
    noop();
    for (auto number : array) {
      noop();
      if (number == 5)
        goto early_exit1;
    }
  }

  do {
    noop();
    do {
      noop();
      goto early_exit1;
    } while (true);
  } while (true);

  do {
    for (auto number : array) {
      noop();
      if (number == 2)
        goto early_exit1;
    }
  } while (true);

  // Jumping further results in error, because the variable declaration would
  // be skipped.
early_exit1:;

  int i = 0;
  while (true) {
    noop();
    while (true) {
      noop();
      if (i > 5)
        goto early_exit2;
      i++;
    }
    noop();
  }

  while (true) {
    noop();
    for (int j = 0; j < 10; ++j) {
      noop();
      if (j > 5)
        goto early_exit2;
    }
    noop();
  }

  while (true) {
    noop();
    for (auto number : array) {
      if (number == 1)
        goto early_exit2;
      noop();
    }
  }

  while (true) {
    noop();
    do {
      noop();
      goto early_exit2;
    } while (true);
  }
early_exit2:;
}

void jump_out_backwards() {

before_the_loop:
  noop();

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      if (i * j > 80)
        goto before_the_loop;
      // CHECK-MESSAGES: [[@LINE-1]]:9: warning: avoid using 'goto' for flow control
      // CHECK-MESSAGES: [[@LINE-8]]:1: note: label defined here
    }
  }
}
