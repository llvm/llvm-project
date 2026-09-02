```{title} clang-tidy - misc-unused-alias-decls
```

# misc-unused-alias-decls

Finds unused namespace alias declarations.

```cpp
namespace my_namespace {
class C {};
}
namespace unused_alias = ::my_namespace;
```
