```{title} clang-tidy - modernize-avoid-setjmp-longjmp
```

# modernize-avoid-setjmp-longjmp

Flags all call expressions involving `setjmp()` and
`longjmp()` in C++ code.

Exception handling with `throw` and `catch` should be used instead.

## References

This check corresponds to the CERT C++ Coding Standard rule
[ERR52-CPP. Do not use setjmp() or longjmp()](https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/exceptions-and-error-handling-err/err52-cpp/).
