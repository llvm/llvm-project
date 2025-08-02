.. title:: clang-tidy - bugprone-suspicious-copy-in-range-loop

bugprone-suspicious-copy-in-range-loop
================================

This check finds ``for`` loops that are potentially-unintentionally copying
the loop variable unnecessarily.

For instance, this check would warn on a loop of the form:

```for (auto s : strings) {
    ...
}
```

This can lead to bugs when the programmer operates on `s` in the loop, assuming
they are mutating the elements directly, but they are in fact only mutating a
temporary copy.

In cases where the programmer's intention is to in fact copy the variable,
the warning can be suppressed by explicitly stating the type.
For instance, this check will *not* warn on a loop of the form

```for (std::string s : strings) {
 ...
 }
 ```

This check is different from `performance-for-range-copy` in that
it triggers on *every* instance where this pattern occurs rather than
potentially-expensive ones.
