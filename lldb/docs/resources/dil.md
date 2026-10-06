# DIL: The Data Inspection Language

This page explains the Data Inspection Language (DIL) in LLDB. Most of this
document is intended for LLDB users, from both the command line and through IDEs
(Integrated Development Environments), Data Formatters, and other scripting
affordances (via the SB APIs). At the end there is some additional information
for LLDB developers.

## Background: What is DIL and why did we implement it?

In the context of LLDB, an **expression path** is the structured syntax used to
navigate and access specific fields, members, or elements within a data
structure starting from a root variable. A **path expression** is an expression
consisting entirely of such structured syntax (and starting from a root
variable).

LLDB has always had two modes for accessing values in your program: path
expressions, that commands like `frame variable` could understand and interpret;
and "other" expressions, which could include expressions that `frame variable`
could handle, but usually also included other pieces written in a source
language, and which could be passed to the `expression` (aka `expr`)
command. These "other" expressions were evaluated using a language-accurate
parser for that language, and the results were obtained by running code in the
target program.

LLDB's path expressions, however, are not necessarily a direct representation of
the type layout of structures in your program. Instead they grew from the
observation that very often the most useful representation of a value is not the
underlying layout of the object (particularly for container classes). In fact,
users who are not data type maintainers are more likely to be confused if shown
the underlying layout of the object. Users generally want to see the semantic
meaning of the object, not its implementation.

LLDB solves this problem by using Data Formatters that take the types in the
type system and produce an alternate layout for the types that correspond to how
the class is used (what the user really wants to see), not how it is
implemented. These re-formatted representations include [Synthetic
Children](https://lldb.llvm.org/use/variable.html#synthetic-children) which are
constructs that are not actually part of the original data type, but which
faciliate showing users what they expect to see. The path expressions give you
access to these re-formatted representations. In addition, these re-formatted
representations allow LLDB to display the dynamic type of an object, not just
the static type. But these re-formatted representations mean nothing to the
underlying source language, so you cannot use them in the expression evaluator
(which is based on the source language). This severely limited the utility of
the reformatted values, since there was no way to perform logic operations (or
other simple expression evaluation) on them.

The Data Inspection Language (DIL) was designed to solve this problem. As the
name implies, the DIL is a language that was created specifically to _improve
the performance_ and _expand the capabilities_ of program data introspection in
LLDB, particularly when that introspection requires evaluating simple
expressions (which is often the case when data formatters and synthetic children
are involved).

The full expression evaluator uses the compiler's language-accurate parser to
fully parse source-language expressions, building up ASTs to represent the
expressions and then JIT'ing code for these ASTs and executing it within the
current LLDB context, by running code in the target. This is a very flexible
mechanism, and can have true source language fidelity, but it can also be a bit
slow (and can also affect the program state).

DIL was explicitly designed for speed. It addresses these issues by avoiding the
compiler altogether. Instead, it defines its own simple grammar, AST
representation, lexer, parser and interpreter. By working on the reformatted
values instead of the raw types, the DIL also allows you to not just view but
write tests and other simple expressions using the values as they are shown to
the user.

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
[dil-expr-lang.ebnf](../../dil-expr-lang.ebnf).


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
situation where an operator has an overloaded definition, DIL will
ignore the overloaded definition, and therefore return a result different from
what the full expression evaluator would return.


Because of this potential for DIL returning a different value than the full
expression evaluator might, we have been cautious about using the full DIL
capabilities in those places in LLDB where we now call DIL before falling back
on the full expression evaluator. For more information on this, see the section
'Programmer settings and changes for controlling or using DIL' below.

## Where and how DIL is used in LLDB

### `frame variable`, `print`, and `expr` commands

LLDB has three main commands that allow users to look at the values of program
data, and to evaluate various types of expressions on them. These three commands
are:

- `frame variable`, aka `frame var` or `v`
- `dwim-print`, aka `print` or `p`
- `expression`, aka `expr`


Historically `frame variable` was intended to handle path expressions (including
re-formatted values), and `expression` was intended to handle any other
expressions users wanted to evaluate. Also `expression` will run code (in the
target) if it needs to, whereas `frame variable` will not. In general this meant
that languages that don't support IR interpretations must always run code in the
target, which will be slow.


`dwim-print` was introduced in 2022.  Before that time, `p` was an alias
for `expr`. The problem was that many LLDB users came to LLDB from GDB, where
users could use one command either for accessing variable values (expression
paths) or for evaluating more complex expressions. The single GDB command was
`print`, usually abbreviated `p`. The result of this was that many LLDB users
would just use `p` all the time, including times when it wasn't really necessary
or even appropriate. `dwim-print` was introduced in an attempt to alleviate this
problem. "dwim" stands for "do-what-I-mean". `dwim-print` looks at the
expression and attempts to decide whether it `frame variable` would produce the
same value as the full expression evaluator, and calls the appropriate mechanism
accordingly. Since `p` was made an alias for `dwim-print`, this went a
long way towards solving the problem of users running code in the target when
that was not necessary.


After being introduced in 2025, DIL became the default implementation for `frame
variable` (aka `v`), so now that command is capable of handling many expressions
that formerly needed to go through the full expression evaluator. This is now
the recommended way for users to request to see the values of their variables,
and to evaluate simple expressions on them.


For full or complex expressions (e.g. things involving function calls or
templates), users should still use the full expression evaluator (`expr`).


Currently`dwim-print`(aka `print` or `p`) dispatches expressions consisting only
of indentifiers and `.` operators to `frame variable`. It sends everything else
to the full expression evaluator. Therefore, to avoid the slow path of running
code in the target (and potentially changing the program state) when it is not
necessary, it is better for users to use either `v` or `expr` to explicitly, and
avoid `dwim-print` or `p`.

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
particularly important. The expression evaluator recognizes this, so it parses
and JIT's the condition only the first time the breakpoint is hit; on subsequent
hits it only has to call a simple function (and run it in the target). Even so,
we have found that just avoiding the overhead of running the code in the target
generally makes DIL interpreted breakpoints ~2.5x faster that expression
evaluated breakpoints.


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
