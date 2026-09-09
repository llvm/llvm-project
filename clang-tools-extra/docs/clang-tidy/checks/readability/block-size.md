```{title} clang-tidy - readability-block-size
```

# readability-block-size

Warns when `if`, `else`, `for` or `while` blocks exceed a given number of lines
of code, and thus flags control flow structures which are likely difficult to
reason about due to their size.

```cpp
// This block will raise a warning if N > IfLineCountThreshold
if (some_condition()){  // line #1
  call_some_fn();       // line #2
  // ...
  do_something_else();  // line #N-1
}                       // line #N
```

## Options

```{option} IfLineCountThreshold

This option sets the number of lines of code beyond which an `if` (or `else`)
block will be flagged as too long. The default value is `20`.
```

```{option} ForLineCountThreshold

This option sets the number of lines of code beyond which a `for` loop will be
flagged as too long. The default value is `30`.
```

```{option} WhileLineCountThreshold

This option sets the number of lines of code beyond which a `while` loop will
be flagged as too long. The default value is `30`.
```
