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

```{option} IgnoreMacros
When `false`, the check also adds missing comments to calls in a macro
argument, such as `ASSERT(f(0))`. It never adds them to calls in a macro body,
or to arguments that you pass through another macro, such as `f(ID(0))`. The
check compares argument comments with parameter names regardless of this
option. Default is `true`.
```

```{option} IgnoreSingleArgument
When `true`, the check will ignore the single argument. Default is `false`.
```

```{option} IgnoredParameterNames
A semicolon-separated list of regular expressions for the parameter names that
the check skips. The check neither compares the argument comment of a matching
parameter nor adds a missing comment for it. A regular expression matches
anywhere in the name unless you anchor it. Use the option for names that say
nothing about the parameter, such as `^[a-z]$;^arg[0-9]*$`. Default is empty.
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
`/*ParameterName=*/` right before the boolean literal argument and the macros
in {option}`BoolLiteralMacros`. Default is `false`.
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
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the integer literal argument.
Default is `false`.
```

Before:

```c++
void foo(int MeaningOfLife);

foo(42);
```

After:

```c++
void foo(int MeaningOfLife);

foo(/*MeaningOfLife=*/42);
```

```{option} CommentNullPtrs
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before the nullptr literal argument and the macros
in {option}`NullPtrMacros`. Default is `false`.
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

```{option} CommentZeroIntegerLiterals
When `true`, the check will add argument comments in the format
`/*ParameterName=*/` right before integer literal arguments with the value
zero, which often stands for "none" or "default". This option has no effect
when {option}`CommentIntegerLiterals` is `true`. Default is `false`.
```

Before:

```c
int open_file(const char *Path, int Flags);

open_file(Path, 0);
open_file(Path, 1);
```

After:

```c
int open_file(const char *Path, int Flags);

open_file(Path, /*Flags=*/0);
open_file(Path, 1);
```

```{option} NullPtrMacros
A semicolon-separated list of macros that {option}`CommentNullPtrs` treats like
`nullptr`. The check only matches an argument that is exactly one of these
macros. Default is `NULL`.
```

```{option} BoolLiteralMacros
A semicolon-separated list of macros that {option}`CommentBoolLiterals` treats
like boolean literals. The check only matches an argument that is exactly one
of these macros. Default is `true;false`, which covers `<stdbool.h>` in C
before C23 and in GNU C++98.
```
