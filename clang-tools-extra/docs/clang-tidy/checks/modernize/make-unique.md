```{title} clang-tidy - modernize-make-unique
```

# modernize-make-unique

This check finds the creation of `std::unique_ptr` objects by explicitly
calling the constructor and a `new` expression, and replaces it with a call
to `std::make_unique`, introduced in C++14.

```c++
auto my_ptr = std::unique_ptr<MyPair>(new MyPair(1, 2));

// becomes

auto my_ptr = std::make_unique<MyPair>(1, 2);
```

This check also finds calls to `std::unique_ptr::reset()` with a `new`
expression, and replaces it with a call to `std::make_unique`.

```c++
my_ptr.reset(new MyPair(1, 2));

// becomes

my_ptr = std::make_unique<MyPair>(1, 2);
```

## Options

(modernize-make-unique-make-smart-ptr-function)=

```{option} MakeSmartPtrFunction
A string specifying the name of make-unique-ptr function. Default is
`std::make_unique`.
```

(modernize-make-unique-make-smart-ptr-function-header)=

```{option} MakeSmartPtrFunctionHeader
A string specifying the corresponding header of make-unique-ptr function.
Default is `<memory>`.
```

(modernize-make-unique-include-style)=

```{option} IncludeStyle
A string specifying which include-style is used, `llvm` or `google`. Default is `llvm`.
```

(modernize-make-unique-ignore-macros)=

```{option} IgnoreMacros
If set to `true`, the check will not give warnings inside macros. Default is `true`.
```

(modernize-make-unique-ignore-default-initialization)=

```{option} IgnoreDefaultInitialization
If set to `false`, the check does not suggest edits that will transform
default initialization into value initialization, as this can cause
performance regressions. Default is `true`.
```
