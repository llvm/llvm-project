// RUN: not %clang_cc1 -std=c++11 -fno-diagnostics-show-line-numbers -fcaret-diagnostics-max-lines=5 -Wsometimes-uninitialized %s 2>&1 | FileCheck %s --strict-whitespace

void line(int);

// Check we expand the range as much as possible within the limit.

// CHECK:      warning: variable 'a' is used uninitialized whenever 'if' condition is true
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}      ^~~~{{$}}
// CHECK-NEXT: note: uninitialized use occurs here
// CHECK-NEXT: {{^}}  return a;
// CHECK-NEXT: {{^}}         ^
// CHECK-NEXT: note: remove the 'if' if its condition is always false
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}  ^~~~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(1);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}  } else {
// CHECK-NEXT: {{^}}  ~~~~~~{{$}}
// CHECK-NEXT: note: initialize the variable
int f1(int cond) {
  int a;
  if (cond) {
    line(1);
  } else {
    a = 3;
  }
  return a;
}

// CHECK:      warning: variable 'a' is used uninitialized whenever 'if' condition is true
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}      ^~~~{{$}}
// CHECK-NEXT: note: uninitialized use occurs here
// CHECK-NEXT: {{^}}  return a;
// CHECK-NEXT: {{^}}         ^
// CHECK-NEXT: note: remove the 'if' if its condition is always false
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}  ^~~~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(1);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(2);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}  } else {
// CHECK-NEXT: {{^}}  ~~~~~~{{$}}
// CHECK-NEXT: note: initialize the variable
int f2(int cond) {
  int a;
  if (cond) {
    line(1);
    line(2);
  } else {
    a = 3;
  }
  return a;
}

// CHECK:      warning: variable 'a' is used uninitialized whenever 'if' condition is true
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}      ^~~~{{$}}
// CHECK-NEXT: note: uninitialized use occurs here
// CHECK-NEXT: {{^}}  return a;
// CHECK-NEXT: {{^}}         ^
// CHECK-NEXT: note: remove the 'if' if its condition is always false
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}  ^~~~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(1);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(2);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(3);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}  } else {
// CHECK-NEXT: {{^}}  ~~~~~~{{$}}
// CHECK-NEXT: note: initialize the variable
int f3(int cond) {
  int a;
  if (cond) {
    line(1);
    line(2);
    line(3);
  } else {
    a = 3;
  }
  return a;
}

// CHECK:      warning: variable 'a' is used uninitialized whenever 'if' condition is true
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}      ^~~~{{$}}
// CHECK-NEXT: note: uninitialized use occurs here
// CHECK-NEXT: {{^}}  return a;
// CHECK-NEXT: {{^}}         ^
// CHECK-NEXT: note: remove the 'if' if its condition is always false
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}  ^~~~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(1);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(2);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(3);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(4);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: note: initialize the variable
int f4(int cond) {
  int a;
  if (cond) {
    line(1);
    line(2);
    line(3);
    line(4);
  } else {
    a = 3;
  }
  return a;
}

// CHECK:      warning: variable 'a' is used uninitialized whenever 'if' condition is true
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}      ^~~~{{$}}
// CHECK-NEXT: note: uninitialized use occurs here
// CHECK-NEXT: {{^}}  return a;
// CHECK-NEXT: {{^}}         ^
// CHECK-NEXT: note: remove the 'if' if its condition is always false
// CHECK-NEXT: {{^}}  if (cond) {
// CHECK-NEXT: {{^}}  ^~~~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(1);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(2);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(3);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: {{^}}    line(4);
// CHECK-NEXT: {{^}}    ~~~~~~~~{{$}}
// CHECK-NEXT: note: initialize the variable
int f5(int cond) {
  int a;
  if (cond) {
    line(1);
    line(2);
    line(3);
    line(4);
    line(5);
  } else {
    a = 3;
  }
  return a;
}


// Check that we don't include lines with no interesting code if we can't reach
// the interesting part within the line limit.

// CHECK:      error: no matching function for call to 'g

// CHECK:      note: candidate template ignored: substitution failure
// CHECK-NEXT: {{^}}decltype(T()
// CHECK-NEXT: {{^}}         ~{{$}}
// CHECK-NEXT: {{^}}    + 1
// CHECK-NEXT: {{^}}    + 2
// CHECK-NEXT: {{^}}    + 3
// CHECK-NEXT: {{^}}void g();
// CHECK-NEXT: {{^}}     ^{{$}}
template<typename T>
decltype(T()
    + 1
    + 2
    + 3)
void g();

// CHECK:      note: candidate template ignored: substitution failure
// CHECK-NEXT: {{^}}void g();
// CHECK-NEXT: {{^}}     ^{{$}}
template<typename T>
decltype(T()
    + 1
    + 2
    + 3
    + 4)
void g();

void h() { g<int()>(); }


void multiple_ranges(int a, int b) {
  // CHECK:      error: invalid operands
  // CHECK-NEXT: &(a)
  // CHECK-NEXT: ~~~~
  // CHECK-NEXT: +
  // CHECK-NEXT: ^
  // CHECK-NEXT: &(b)
  // CHECK-NEXT: ~~~~
  &(a)
  +
  &(b);

  // CHECK-NEXT: error: invalid operands
  // CHECK-NEXT: &(
  // CHECK-NEXT: ~~
  // CHECK-NEXT: a
  // CHECK-NEXT: ~
  // CHECK-NEXT: )
  // CHECK-NEXT: ~
  // CHECK-NEXT: +
  // CHECK-NEXT: ^
  // CHECK-NEXT: &(
  // CHECK-NEXT: ~~
  &(
  a
  )
  +
  &(
  b
  );

  // CHECK-NEXT: error: invalid operands
  // CHECK-NEXT: &(a
  // CHECK-NEXT: ~~
  // CHECK-NEXT: )
  // CHECK-NEXT: ~
  // CHECK-NEXT: +
  // CHECK-NEXT: ^
  // CHECK-NEXT: &(
  // CHECK-NEXT: ~~
  // CHECK-NEXT: b
  // CHECK-NEXT: ~
  &(a
  )
  +
  &(
  b
  );
}

#define pr33902_a(b) #b
#define pr33902_c(d) _Pragma(pr33902_a(d))
#define pr33902_e(f) pr33902_c(GCC warning #f)
pr33902_e() pr33902_e()
