Cilk-Clang
================================

This version of Clang supports the `_Cilk_spawn`, `_Cilk_sync`, and
`_Cilk_for` keywords from Cilk.  In particular, this version of Clang
supports the use of _Cilk_spawn before a function call in a statement,
an assignment, or a declaration, as in the following examples:

```
_Cilk_spawn foo(n);
```

```
x = _Cilk_spawn foo(n);
```

```
int x = _Cilk_spawn foo(n);
```

When spawning a function call, the call arguments and function
arguments are evaluated before the spawn occurs.  When spawning an
assignment or declaration, the LHS is also evaluated before the spawn
occurs.

For convenience, this version of Clang allows `_Cilk_spawn` to spawn an
arbitrary statement, as follows:

```
_Cilk_spawn { x = foo(n); }
```

Please use this syntax with caution!  When spawning an arbitrary
statement, the spawn occurs before the evaluation of any part of the
spawned statement.  Furthermore, some statements, such as `goto`, are
not legal to spawn.  In the future, we will add checks to catch
illegal uses of `_Cilk_spawn`.

