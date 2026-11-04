# clang-tidy - llvm-invalid-regex-pattern

## llvm-invalid-regex-pattern

Detects malformed regex patterns defined in a single string literal.
It detects these string literals if they are defined in the regex constructor
with a string literal, or stored in one of these non mutable containers:

- `const std::string`
- `const char*`
- `const char[]`
- `static const char[]`
- `const llvm::StringRef`
- `std::string_view`

In the event that the patterns are stored as a class member, the check verifies
the initialization value, if defined, even if its overwritten by the constructor.

Example of detection:

```cpp
const std::string my_regex_pattern("[0-8"); // invalid regex pattern
llvm::Regex my_regex(my_regex_pattern);
```

Example of member data being overwritten:

```cpp
class foo{
public:
  foo(llvm::StringRef regex_pat) : regex_p(regex_pat){};
  const llvm::StringRef regex_pat = "("; // invalid regex pattern
};
foo bar("[0-9]"); // pattern not checked
llvm::Regex my_regex(bar.regex_pat);
```
