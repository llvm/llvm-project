```{title} clang-tidy - bugprone-copy-constructor-mutates-argument
```

# bugprone-copy-constructor-mutates-argument

Finds assignments to the copied object and its direct or indirect members
in copy constructors and copy assignment operators.

This check corresponds to the CERT C Coding Standard rule
[OOP58-CPP. Copy operations must not mutate the source object][cert-oop58-cpp].

[cert-oop58-cpp]: https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/object-oriented-programming-oop/oop58-cpp/
