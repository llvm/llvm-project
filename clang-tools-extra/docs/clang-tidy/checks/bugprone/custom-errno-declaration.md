# bugprone-custom-errno-declaration

Finds custom declarations of `extern int` variable named `errno`.
It is able to fix the problem by removing the line of the declaration and inserting `#include <cerrno>`
at the top of the file.

For further reading, see [the page of SEI CERT C Coding Standard]
(https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/miscellaneous-msc/msc38-c/).

Example:

```cpp
extern int errno;
```
