// RUN: %check_clang_tidy %s bugprone-inconsistent-ifelse-braces %t

void foo(bool flag) {
  int x, y;

  if (flag)
    x = 0;
  else {
    y = 1;
  }

  if (flag) {
    x = 2;
  } else {
    y = 3;
  }

  if (flag) {
    x = 4;
  } else if (flag) {
    x = 5; 
  } else {
    y = 6;
  }

  if (flag)
    x = 7;
  else if (flag)
    y = 8;
  else
    y = 9;

  if (flag)
    x = 10;
  else if (flag) {
    y = 11;
  } else
    y = 12;
}

// FIXME: Add something that triggers the check here.
void f();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'f' is insufficiently awesome [bugprone-inconsistent-ifelse-braces]

// FIXME: Verify the applied fix.
//   * Make the CHECK patterns specific enough and try to make verified lines
//     unique to avoid incorrect matches.
//   * Use {{}} for regular expressions.
// CHECK-FIXES: {{^}}void awesome_f();{{$}}

// FIXME: Add something that doesn't trigger the check here.
void awesome_f2();
