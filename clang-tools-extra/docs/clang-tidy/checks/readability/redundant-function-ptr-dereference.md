```{title} clang-tidy - readability-redundant-function-ptr-dereference
```

# readability-redundant-function-ptr-dereference

Finds redundant dereferences of a function pointer.

Before:

```c++
int f(int,int);
int (*p)(int, int) = &f;

int i = (**p)(10, 50);
```

After:

```c++
int f(int,int);
int (*p)(int, int) = &f;

int i = (*p)(10, 50);
```

