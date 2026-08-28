```{title} clang-tidy - portability-avoid-pragma-comment
```

# portability-avoid-pragma-comment

Finds uses of `#pragma comment` and, for `lib` or `linker` comments, suggests
using the build system for improved portability.

`#pragma comment` is not widely supported outside of MSVC. Clang supports the
use of `#pragma comment` to link libraries on both Windows and Linux, but other
kinds are only supported on Windows. Using `pragma comment` to change link flags
may be unexpected in projects that prefer to set these flags in the build
system.

```c++
// Clang supports the `lib` kind on Windows and Linux, but setting link flags
// outside the build system may be unexpected
#pragma comment(lib, "some_lib")
#pragma comment(linker, "some_linker_flag")

// Clang only supports the `compiler` and `user` kinds when targeting Windows
#pragma comment(compiler)
#pragma comment(user, "Some string")
```
