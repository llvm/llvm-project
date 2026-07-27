```{title} clang-tidy - bugprone-nested-switch-label
```

# bugprone-nested-switch-label

Finds `case` and `default` labels nested in compound statements other than
the compound statement that forms the body of their switch. Such labels are
legal, but their control flow is easy to misread because the switch can jump
directly into the nested scope.

For example:

```cpp
switch (value) {
case 0:
  if (condition) {
    first();
    break;
  case 1: // Warning: entering here bypasses the condition and first().
    second();
    break;
  }
  break;
default:
  break;
}
```
