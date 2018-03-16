.. title:: clang-tidy - bugprone-macro-parentheses

bugprone-macro-parentheses
==========================


Finds macros that can have unexpected behaviour due to missing parentheses.

Macros are expanded by the preprocessor as-is. As a result, there can be
unexpected behaviour; operators may be evaluated in unexpected order and
unary operators may become binary operators, etc.

When the replacement list has an expression, it is recommended to surround
it with parentheses. This ensures that the macro result is evaluated
completely before it is used.

It is also recommended to surround macro arguments in the replacement list
with parentheses. This ensures that the argument value is calculated
properly.
