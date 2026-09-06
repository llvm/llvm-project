```{title} clang-tidy - readability-function-size
```

# readability-function-size

`google-readability-function-size` redirects here as an alias for this check.

Checks for large functions based on various metrics.

## Options

```{option} LineThreshold
Flag functions exceeding this number of lines. Default is `none` (ignore the
number of lines).
```

```{option} StatementThreshold
Flag functions exceeding this number of statements. This may differ
significantly from the number of lines for macro-heavy code. Default is
`800`.
```

```{option} BranchThreshold
Flag functions exceeding this number of control statements. Default is
`none` (ignore the number of branches).
```

```{option} ParameterThreshold
Flag functions that exceed a specified number of parameters. Default
is `none` (ignore the number of parameters).
```

```{option} NestingThreshold
Flag compound statements which create next nesting level after
{option}`NestingThreshold`. This may differ significantly from the expected
value for macro-heavy code. Default is `none` (ignore the nesting level).
```

```{option} VariableThreshold
Flag functions exceeding this number of variables declared in the body.
Please note that function parameters and variables declared in lambdas,
GNU Statement Expressions, and nested class inline functions are not counted.
Default is `none` (ignore the number of variables).
```

```{option} CountMemberInitAsStmt
When `true`, count class member initializers in constructors as statements.
Default is `true`.
```

```{option} IgnoreMacros
When `true`, the check will not count statements, branches, nesting
levels, or variable declarations inside macros. Default is `false`.
```
