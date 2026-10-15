```{title} clang-tidy - modernize-make-shared
```

# modernize-make-shared

This check finds the creation of `std::shared_ptr` objects by explicitly
calling the constructor and a `new` expression, and replaces it with a call
to `std::make_shared`.

```c++
auto my_ptr = std::shared_ptr<MyPair>(new MyPair(1, 2));

// becomes

auto my_ptr = std::make_shared<MyPair>(1, 2);
```

This check also finds calls to `std::shared_ptr::reset()` with a `new`
expression, and replaces it with a call to `std::make_shared`.

```c++
my_ptr.reset(new MyPair(1, 2));

// becomes

my_ptr = std::make_shared<MyPair>(1, 2);
```

## Options

(modernize-make-shared-make-smart-ptr-function)=

```{option} MakeSmartPtrFunction
A string specifying the name of make-shared-ptr function. Default is
`std::make_shared`.
```

(modernize-make-shared-make-smart-ptr-function-header)=

```{option} MakeSmartPtrFunctionHeader
A string specifying the corresponding header of make-shared-ptr function.
Default is `<memory>`.
```

(modernize-make-shared-include-style)=

```{option} IncludeStyle
A string specifying which include-style is used, `llvm` or `google`. Default is `llvm`.
```

(modernize-make-shared-ignore-macros)=

```{option} IgnoreMacros
If set to `true`, the check will not give warnings inside macros. Default is `true`.
```

(modernize-make-shared-ignore-default-initialization)=

```{option} IgnoreDefaultInitialization
If set to `false`, the check does not suggest edits that will transform
default initialization into value initialization, as this can cause
performance regressions. Default is `true`.
```
