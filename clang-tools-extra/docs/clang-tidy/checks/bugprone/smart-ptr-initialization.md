# clang-tidy - bugprone-smart-ptr-initialization

## bugprone-smart-ptr-initialization

Detects dangerous initialization of smart pointers with raw pointers that are
already owned elsewhere, which can lead to double deletion.

This check implements CERT C++ rule "MEM56-CPP. Do not store an already-owned
pointer value in an unrelated smart pointer".

## Example

The check flags cases where raw pointers that are already owned or managed
elsewhere are passed to smart pointer constructors or `reset()` methods:

```cpp
#include <memory>

void f() {
  int *i = new int;
  std::shared_ptr<int> p1(i);
  // Warning: 'i' is already managed by another smart pointer
  std::shared_ptr<int> p2(i);
}
```

## Options

- **SharedPointers**
  
  A semicolon-separated list of (fully qualified) shared pointer type names
  that should be checked. Default value is
  `::std::shared_ptr;::boost::shared_ptr`.

- **UniquePointers**
  
  A semicolon-separated list of (fully qualified) unique pointer type names
  that should be checked. Default value is `::std::unique_ptr`.

- **DefaultDeleters**
  
  A semicolon-separated list of (fully qualified) default deleter type names.
  Smart pointers with deleters matching these types are considered to use the
  default deleter and are checked. Smart pointers with custom deleters are
  ignored. Default value is `::std::default_delete`.

- **StrictMode**
  
  When enabled, the check only allows raw pointers from `new` expressions or
  `std::unique_ptr::release()` to be passed to smart pointer constructors or
  `reset()` methods. Any other raw pointer source (function returns, pointers
  from containers, etc.) will trigger a warning. Default value is `false`.  
  **Note**: This mode produces a significant number of false positives, as it
  cannot reliably distinguish between owning and non-owning raw pointers in
  complex code. It is **not recommended** for legacy codebases. Consider
  enabling this option only for new projects written with strict ownership
  policies from the start.
  **Note**: Smart pointers with custom deleters (i.e., deleters not listed in
  the `DefaultDeleters` option) are always ignored in this mode and will never
  produce any diagnostics, regardless of the initialization pattern.

## Limitations

This check only supports smart pointers with shared and unique ownership
semantics. Smart pointers with different semantics, such as
`boost::scoped_ptr`, cannot be used with the current version of this check.

This check unable to catch relevant cases inside a ternary operator:

   ```cpp
   std::shared_ptr<A> a(flag ? nullptr : &getA());
   ```

The warning will never be shown with the current version of this check.

## References

- [CERT C++ MEM56-CPP](https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM56-CPP.+Do+not+store+an+already-owned+pointer+value+in+an+unrelated+smart+pointer)
- [C++ Core Guidelines R.3: A raw pointer (a T*) is non-owning](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r3-a-raw-pointer-a-t-is-non-owning)
- [C++ Core Guidelines R.20: Use unique_ptr or shared_ptr to represent ownership](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r20-use-unique_ptr-or-shared_ptr-to-represent-ownership)