```{title} clang-tidy - readability-redundant-control-flow
```

# readability-redundant-control-flow

This check looks for procedures (functions returning no value) with
`return` statements at the end of the function. Such `return` statements
are redundant.

Loop statements (`for`, `while`, `do while`) are checked for redundant
`continue` statements at the end of the loop body.

The check also follows nested blocks and the branches of a final `if`
statement. It preserves exits that skip subsequent statements, returns
inside loops, and return expressions with side effects. For an unbraced
branch or loop body, removing the keyword leaves a null statement.

Examples:

The following function `f` contains a redundant `return` statement:

```c++
extern void g();
void f() {
  g();
  return;
}
```

becomes

```c++
extern void g();
void f() {
  g();
}
```

The following function `k` contains a redundant `continue` statement:

```c++
void k() {
  for (int i = 0; i < 10; ++i) {
    continue;
  }
}
```

becomes

```c++
void k() {
  for (int i = 0; i < 10; ++i) {
  }
}
```
