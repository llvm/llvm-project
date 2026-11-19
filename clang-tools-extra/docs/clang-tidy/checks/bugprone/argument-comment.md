```{title} clang-tidy - bugprone-argument-comment
```

# bugprone-argument-comment

Checks that argument comments match parameter names and can optionally add
missing comments for literals, init-lists, and constructed temporaries.

The check understands argument comments in the form `/*parameter_name=*/`
that are placed right before the argument.

```c++
void f(bool foo);

...

f(/*bar=*/true);
// warning: argument name 'bar' in comment does not match parameter name 'foo'
```

The check tries to detect typos and suggest automated fixes for them. It can
also insert missing comments for configured argument kinds.

## Options

```{option} StrictMode
When `false`, the check will ignore leading and trailing
underscores and case when comparing names -- otherwise they are taken into
account. Default is `false`.
```

```{option} IgnoreSingleArgument
When `true`, the check will ignore the single argument. Default is `false`.
```

```{option} IgnoredParameterNames
A semicolon-separated list of regular expressions for the parameter names that
the check skips. The check neither compares the argument comment of a matching
parameter nor adds a missing comment for it. A regular expression matches
anywhere in the name unless you anchor it. Use the option for names that say
nothing about the parameter, such as `^[a-z]$;^arg[0-9]*$`. Default is an
empty string.
```

```{option} CommentAnonymousInitLists
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before anonymous braced-init list arguments
such as `{}` and `{1, 2, 3}`. Default is `false`.
```

Before:

```c++
void foo(const std::vector<int> &Dims);

foo({});
```

After:

```c++
void foo(const std::vector<int> &Dims);

foo(/*Dims=*/{});
```

```{option} CommentBoolLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the boolean literal argument.
Default is `false`.
```

Before:

```c++
void foo(bool TurnKey, bool PressButton);

foo(true, false);
```

After:

```c++
void foo(bool TurnKey, bool PressButton);

foo(/*TurnKey=*/true, /*PressButton=*/false);
```

```{option} CommentCharacterLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the character literal argument.
Default is `false`.
```

Before:

```c++
void foo(char *Character);

foo('A');
```

After:

```c++
void foo(char *Character);

foo(/*Character=*/'A');
```

```{option} CommentFloatLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the float/double literal argument.
Default is `false`.
```

Before:

```c++
void foo(float Pi);

foo(3.14159);
```

After:

```c++
void foo(float Pi);

foo(/*Pi=*/3.14159);
```

```{option} CommentIntegerLiterals
When `All`, the check will add argument comments in the format
`/*ParameterName=*/` right before every integer literal argument. When
`ZeroOnly`, it adds them before integer literal arguments with the value zero,
such as `0`, `0x0` and `0UL`, which often stand for "none" or "default". When
`None`, it adds no comments before integer literal arguments. The values `true`
and `false` mean `All` and `None`. Default is `None`.
```

Before:

```c++
void foo(int MeaningOfLife);

foo(42);
```

After, with `All`:

```c++
void foo(int MeaningOfLife);

foo(/*MeaningOfLife=*/42);
```

Before:

```c
int open_file(const char *Path, int Flags);

open_file(Path, 0);
open_file(Path, 1);
```

After, with `ZeroOnly`:

```c
int open_file(const char *Path, int Flags);

open_file(Path, /*Flags=*/0);
open_file(Path, 1);
```

```{option} CommentNullPtrs
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the nullptr literal argument.
Default is `false`.
```

Before:

```c++
void foo(A* Value);

foo(nullptr);
```

After:

```c++
void foo(A* Value);

foo(/*Value=*/nullptr);
```

```{option} CommentParenthesizedTemporaries
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before explicit temporary constructions such as
`Type()` and `Type(1, 2, 3)`. Default is `false`.
```

Before:

```c++
struct Dims {
  Dims();
  Dims(int, int, int);
};

void foo(const Dims &DimsValue);

foo(Dims());
foo(Dims(1, 2, 3));
```

After:

```c++
struct Dims {
  Dims();
  Dims(int, int, int);
};

void foo(const Dims &DimsValue);

foo(/*DimsValue=*/Dims());
foo(/*DimsValue=*/Dims(1, 2, 3));
```

```{option} CommentStringLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the string literal argument.
Default is `false`.
```

Before:

```c++
void foo(const char *String);
void foo(const wchar_t *WideString);

foo("Hello World");
foo(L"Hello World");
```

After:

```c++
void foo(const char *String);
void foo(const wchar_t *WideString);

foo(/*String=*/"Hello World");
foo(/*WideString=*/L"Hello World");
```

```{option} CommentTypedInitLists
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before typed braced-init list arguments such
as `Type{}`. Default is `false`.
```

Before:

```c++
void foo(const std::vector<int> &Dims);

foo(std::vector<int>{});
```

After:

```c++
void foo(const std::vector<int> &Dims);

foo(/*Dims=*/std::vector<int>{});
```

```{option} CommentUserDefinedLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the user defined literal argument.
Default is `false`.
```

Before:

```c++
void foo(double Distance);

double operator"" _km(long double);

foo(402.0_km);
```

After:

```c++
void foo(double Distance);

double operator"" _km(long double);

foo(/*Distance=*/402.0_km);
```
