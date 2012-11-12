// RUN: %clang_cc1 -ferror-limit 1 -fsyntax-only %s 2>&1 | FileCheck %s

// error and three notes emitted
void foo(int);
void foo(double);
void foo(int, int);

int main()
{
    foo();
}

// error and note suppressed by error-limit
struct s1{};
struct s1{};

// CHECK: 10:5: error: no matching function for call to 'foo'
// CHECK: 6:6: note: candidate function not viable: requires 2 arguments, but 0 were provided
// CHECK: 5:6: note: candidate function not viable: requires 1 argument, but 0 were provided
// CHECK: 4:6: note: candidate function not viable: requires 1 argument, but 0 were provided
// CHECK: fatal error: too many errors emitted, stopping now
// CHECK-NOT: 15:8: error: redefinition of 's1'
// CHECK-NOT: 14:8: note: previous definition is here
