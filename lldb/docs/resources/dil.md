# DIL: The Data Inspection Language

This page explains the Data Inspection Language (DIL) in LLDB. Most of this
document is intended for LLDB users, from both the command line and through
IDEs (Integrated Development Environments) via lldb-dap. At the end there is
some additional information for LLDB developers.

## Background: What is DIL and why did we implement it?

As the name implies, the Data Inspection Language is a language that was created
specifically to improve the stability and performance of program data
introspection in LLDB, particularly when that introspection requires evaluating
any simple expressions (which is often the case when data formatters and
synthetic children are involved).

Prior to the introduction of DIL, nearly all expression evaluations in LLDB had
to be done through the full expression evaluator. This is a heavy-weight
mechanism that uses Clang to fully parse C++ expressions, building up Clang ASTs
to represent the expressions and then evaluating these Clang ASTs within the
current LLDB context. This approach has historically had several drawbacks,
including being somewhat slow and also sometimes crashing (mostly due to Clang
being designed to expect complete, correct programs rather than partial contexts
and expressions).

DIL was designed to address these issues by avoiding Clang altogether. Instead,
it defines its own simple grammar, AST representation, lexer, parser and
interpreter. By bypassing Clang, and only handling comparatively simple
expressions, DIL can be both faster and more stable than the full expression
evaluator.


## The DIL Language

First and foremost, it is important to understand that DIL is **its own
language**.

While it is very similar to C-based languages, DIL is NOT C, nor C++, nor any
other specific programming language. Deciding to make DIL its own language, for
the specific purpose of helping introspect program data in LLDB, allows us to
both simplify the language (for example, we do not need to support all the
complexities of C++), and to easily support multiple programming languages -- we
are free to add features outside of standard C/C++ that might facilitate
supporting such languages as Rust, Swift or Fortran.

The actual formal definition of DIL can be found in an EBNF file in the LLDB
source code repository in
[dil-expr-lang.ebnf](https://github.com/llvm/llvm-project/blob/main/lldb/docs/dil-expr-lang.ebnf).


Here is a quick summary of the types of expressions DIL can support:

- Identifier names, boolean values, numbers, register names, `nullptr`
- Logical-and, logical-or, logical-not: `&&`, `||`, `!`
- Bitwise-and, exclusive-or, inclusive-or: `&`, `^`, `|`
- Relational and equality expressions: `<`, `>`, `>=`, `<=`, `==`, `!=`
- Shift expressions: `>>`, `<<`
- Basic arithmetic expressions: `+`, `-`, `*`, `/`, `%`
- Array/Vector indexing expression: `[]`
- Member-of, both via `.` and `->` (as appropriate)
- Pointer dereferencing: `->`, `*`
- Address-of: `&`
- Bitwise Not: `~`
- Type casting, using C-Style syntax, for builtin types, class names, enum names and typedef names
- Assignment and Composite Assignment: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- Parenthesized expressions: `(`, `)`
- Ternary conditional operator: `?` `:`

NOTE: DIL only allows ONE assignment of any kind in any given expression. This
is because with multiple assignments it becomes impossible to guarantee a
deterministic evaluation order, which could have unexpected consequences (think
about `a=0; a++ + a++ + a++` -- what should that return?).

## CAUTION: DIL vs. Expression Evaluator

As we took pains to point out above, DIL is NOT identical to C/C++. Therefore,
you cannot count on it behaving identically to C/C++.

DIL, and those places in LLDB where it is used to evaluate expressions, can now
handle many expressions that only the full expression evaluator could handle in
the past.

Most of the time, this is not a problem, as the two different execution paths
will usually return the same value for the same expression. HOWEVER, there are
cases where they will both return apparently valid but DIFFERENT RESULTS. One
example of this is in the case of operator overloading. If DIL is used in a
situation where an operator has an overloaded definition, DIL will probably
ignore the overloaded definition, and therefore return a result different from
what you might expect. The full expression evaluator handles operator
overloading properly.

Because of this potential for DIL returning a different value than the full
expression evaluator might, we have been cautious about using the full DIL
capabilities in those places in LLDB where we now call DIL before falling back
on the full expression evaluator. For more information on this, see the section
'Programmer settings and changes for controlling or using DIL' below.

## Where and how DIL is used in LLDB

### Historical Background: `frame variable`, `print`, and `expr` commands

The LLDB interactive commands for examining program variables can be rather
confusing. Currently there are three main commands that allow users to look at
the values of program data, and to evaluate various types of expressions on
them. These three commands are:

- `frame variable`, aka `frame var` or `v`
- `dwim-print`, aka `print` or `p`
- `expr`

As if that were not confusing enough, `dwim-print` did not exist
until 2022. Before that, `print` and `p` were abbreviations for `expr`.

So what do these various commands do, and how did we get into this confusing
mess?

Originally, `frame variable` was meant to allow only plain access to
variables. It recognized and handled a few basic operators as part of this:
Address-of, pointer dereferencing, finding fields/members, and vector/array
indexing. Any more complicated expression evaluation was meant to go through the
expression evaluator. The expectation was that users would use `v` for simple
accesses and `p` for more complex evaluations.

The problem was that nearly all the users who came to LLDB were coming from using
GDB. GDB did not have two separate commands -- it used `p` (`print`) for
everything. So LLDB users were almost always using `p`, which was sometimes
surprisingly slow or even crashed (see Background above, for more information).

In an attempt to help properly direct more of the calls that should be going
through `frame variable`, `dwim-print` was introduced in 2022. 'dwim' stands for
'do-what-I-mean'. `dwim-print` looks at the expression and attempts to decide
whether it could/should be handled by `frame variable` or whether it really
needs the full expression evaluator, and calls the appropriate mechanism
accordingly.

DIL is now the default implementation for `frame variable` (aka `v`), so now
that command is capable of handling many expressions that formerly needed to go
through the full expression evaluator.

`dwim-print` (aka `print` or `p`) still dispatches expressions based on what the
original `frame variable` implementation could handle, not what DIL can do.


### `frame variable`, `print` and `expr` commands today

DIL is now the default implementation underlying the `frame variable` command
(`v` for short). This is now the recommended way for users to request to see the
values of their variables, and to evaluate simple expressions on them.

For full or complex expressions (e.g. things involving function calls or
templates), users should still use the full expression evaluator (`expr`).

Using `dwim-print`, `print` or `p` still works, and will dispatch things as it
always has (old `frame var` things to `frame var` and everything else to
`expr`). However, users should probably use either `v` or `expr` to explicitly
choose the path they really want.


### User options and flags to control using DIL

#### target.experimental.use-DIL

Whether DIL is actually used when `frame variable` is called is controlled by a
Boolean setting, `target.experimental.use-DIL`. This setting defaults to
`true`. Users can set this to `false`, which will cause the `frame variable`
command to fall back onto its old implementation:

```
(lldb) v i
(int) i = 0
(lldb) v 'i+3'
(int) result = 3
(lldb) settings show target.experimental.use-DIL
target.experimental.use-DIL (boolean) = true
(lldb) settings set target.experimental.use-DIL false
(lldb) v 'i+3'
error: unexpected char '+' encountered after "i" in "+3"
(lldb) 
```

#### target.breakpoints-condition-mode

Evaluating expressions for conditional breakpoints is one place where speed is
particularly important, so using DIL to evaluate these conditions whenever
possible is probably desirable.

On the other hand, as mentioned in the 'CAUTION: DIL vs. Expression Evaluator'
section above, DIL can occasionally return different results than would have
been obtained by calling the full expression evaluator.

We have added a new target setting to LLDB,
`target.breakpoints-condition-mode`, with three valid values: `dil`, `expr`, and
`dwim`. You can see more information about these modes by typing `help
condition-mode` at the LLDB prompt. However, they are pretty much what you would
expect: `dil` uses the DIL expression parser and interpreter to handle all
breakpoint conditions; `expr` uses the full expression evaluator to handle all
breakpoint conditions; and `dwim` will choose between the two depending on the
expression.

```
(lldb) settings show target.breakpoints-condition-mode
target.breakpoints-condition-mode (enum) = dwim
(lldb) help condition-mode
  <condition-mode> -- Specifies the mode to use when evaluating the condition expression of breakpoints.

     dil  : Use Data Inspection Language (DIL) to evaluate the condition.
     expr : Use UserExpression to evaluate the condition.
     dwim : Use DIL to evaluate the condition, and if it fails, fall back to UserExpression.

(lldb) 
```

You can also choose the breakpoint condition mode when setting a condition on a
particular breakpoint by using the `--condition-mode` (`-Z`) flag and specifying
the mode (same three options as above).


#### target.experimental.use-DIL-for-creating-values

The Boolean setting `target.experimental.use-DIL-for-creating-values` (which
defaults to `true`) is used in `SBValue::CreateValueFromExpression` to control
whether to try evaluating the expression with DIL first, or to bypass DIL and
go straight to the full expression evaluator.

### Programmer settings and changes for controlling or using DIL

#### DILMode

The main "knob" we have created for controlling the behavior of DIL
programmatically is via the LLDB enumeration `DILMode`, defined in the
lldb-enumerations.h file. There are three DILMode values:

- eDILModeSimple, which handles only identifiers and the `.` operator.
- eDILModeLegacy, which handles what the old `frame variable` implementation handled, i.e. identifiers, integers, `.`, `->`, `*`, `&`, and `[]`.
- eDILModeFull, which handles everything supported by DIL.


#### TryDILFirst in EvaluateExpressionOptions, SBExpressionOptions

We have added a new Boolean private member, `m_try_DIL_first`, to the
EvaluateExpressionOptions class, along with the public functions
`GetTryDILFirst` and `SetTryDILFirst`. The default value for `m_try_DIL_first`
is `false`.

`SBExpressionOptions` has corresponding `GetTryDILFirst` and `SetTryDILFirst`
functions, which end up calling the EvaluateExpressionOptions functions.

These methods are called from two different versions of
`SBValue::CreateValueFromExpression` (which is an overloaded function, hence the
multiple versions). One version calls `SetTryDILFirst`, passing the value from
the setting `target.experimental.use-DIL-for-creating-values`. The other version
calls `GetTryDILFirst`, and acts appropriately, either calling DIL first
(falling back on the expression evaluator if necessary), or bypassing DIL and
going straight to the expression evaluator.

#### SBFrame::GetValueForVariablePathWithMode

Initially we wanted to update SBFrame::GetValueForVariablePath directly to use
DIL, controlled by a DILMode parameter. However, that involved making a breaking
change to the LLDB API. So instead we added two new functions to SBFrame:

```
lldb::SBValue GetValueForVariablePathWithMode(const char *var_path,
                                              lldb::DILMode mode,
                                              DynamicValueType use_dynamic);

lldb::SBValue GetValueForVariablePathWithMode(const char *var_path,
                                              lldb::DILMode mode);

```

As you might guess, they are similar to SBFrame::GetValueForVariablePath, but
they use DIL and allow explicitly setting which DILMode to use, rather than 
using the default (eDILModeFull).


#### Changes to lldb-dap

There are several places in lldb-dap that used to directly call the full
expression evaluator, and which we thought might benefit from trying to call DIL
first. So we updated the following functions to do exactly that, by calling
SBFrame::GetValueForVariablePathWithMode first, and falling back on the full
expression evaluator if that failed:

- EvaluateVariableExpression, in EvaluateRequestHandler.cpp
- SourceBreakpoint::BreakpointHitCallback, in SourceBreakpoint.cpp
