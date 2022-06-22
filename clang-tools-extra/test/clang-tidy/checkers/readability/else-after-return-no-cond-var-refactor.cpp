// RUN: %check_clang_tidy %s readability-else-after-return %t -- \
// RUN:     -config='{CheckOptions: [ \
// RUN:         {key: readability-else-after-return.WarnOnConditionVariables, value: false}, \
// RUN:     ]}'

bool foo(int Y) {
  // Excess scopes are here so that the check would have to opportunity to
  // refactor the variable out of the condition.

  // Expect warnings here as we don't need to move declaration of 'X' out of the
  // if condition as its not used in the else.
  {
    if (int X = Y)
      return X < 0;
    else
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
      return false;
  }
  {
    if (int X = Y; X)
      return X < 0;
    else
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use 'else' after 'return'
      return false;
  }

  // Expect no warnings for these cases, as even though its safe to move
  // declaration of 'X' out of the if condition, that has been disabled
  // by the options.
  {
    if (int X = Y)
      return false;
    else
      return X < 0;
  }
  {
    if (int X = Y; X)
      return false;
    else
      return X < 0;
  }
}
