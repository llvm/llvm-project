<!--===- docs/ParserCombinators.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Parser Combinators

```eval_rst
.. contents::
   :local:
```

This document is a primer on Parser Combinators and their use in Flang.

## Concept
The Fortran language recognizer here can be classified as an LL recursive
descent parser.  It is composed from a *parser combinator* library that
defines a few fundamental parsers and a few ways to compose them into more
powerful parsers.

For our purposes here, a *parser* is any object that attempts to recognize
an instance of some syntax from an input stream.  It may succeed or fail.
On success, it may return some semantic value to its caller.

In C++ terms, a parser is any instance of a class that
1. has a `constexpr` default constructor,
1. defines a type named `resultType`, and
1. provides a function (`const` member or `static`) that accepts a reference to a
`ParseState` as its argument and returns a `std::optional<resultType>` as a
result, with the presence or absence of a value in the `std::optional<>`
signifying success or failure, respectively.
```
std::optional<resultType> Parse(ParseState &) const;
```
The `resultType` of a parser is typically the class type of some particular
node type in the parse tree.

`ParseState` is a class that encapsulates a position in the source stream,
collects messages, and holds a few state flags that determine tokenization
(e.g., are we in a character literal?).  Instances of `ParseState` are
independent and complete -- they are cheap to duplicate whenever necessary to
implement backtracking.

The `constexpr` default constructor of a parser is important.  The functions
(below) that operate on instances of parsers are themselves all `constexpr`.
This use of compile-time expressions allows the entirety of a recursive
descent parser for a language to be constructed at compilation time through
the use of templates.

### Fundamental Predefined Parsers
These objects and functions are (or return) the fundamental parsers:

* `ok` is a trivial parser that always succeeds without advancing.
* `pure(x)` returns a trivial parser that always succeeds without advancing,
  returning some value `x`.
* `pure<T>()` is `pure(T{})` but does not require that T be copy-constructible.
* `fail<T>(msg)` denotes a trivial parser that always fails, emitting the
  given message as a side effect.  The template parameter is the type of
  the value that the parser never returns.
* `nextCh` consumes the next character and returns its location,
  and fails at EOF.
* `"xyz"_ch` succeeds if the next character consumed matches any of those
  in the string and returns its location.  Be advised that the source
  will have been normalized to lower case (miniscule) letters outside
  character and Hollerith literals and edit descriptors before parsing.

### Combinators
These functions and operators combine existing parsers to generate new parsers.
They are `constexpr`, so they should be viewed as type-safe macros.

* `!p` succeeds if p fails, and fails if p succeeds.
* `p >> q` fails if p does, otherwise running q and returning its value when
  it succeeds.
* `p / q` fails if p does, otherwise running q and returning p's value
  if q succeeds.
* `p || q` succeeds if p does, otherwise running q.  The two parsers must
  have the same type, and the value returned by the first succeeding parser
  is the value of the combination.
* `first(p1, p2, ...)` returns the value of the first parser that succeeds.
  All of the parsers in the list must return the same type.
  It is essentially the same as `p1 || p2 || ...` but has a slightly
  faster implementation and may be easier to format in your code.
* `lookAhead(p)` succeeds if p does, but doesn't modify any state.
* `attempt(p)` succeeds if p does, safely preserving state on failure.
* `many(p)` recognizes a greedy sequence of zero or more nonempty successes
  of p, and returns `std::list<>` of their values.  It always succeeds.
* `some(p)` recognized a greedy sequence of one or more successes of p.
  It fails if p immediately fails.
* `skipMany(p)` is the same as `many(p)`, but it discards the results.
* `maybe(p)` tries to match p, returning an `std::optional<T>` value.
  It always succeeds.
* `defaulted(p)` matches p, and when p fails it returns a
  default-constructed instance of p's resultType.  It always succeeds.
* `nonemptySeparated(p, q)` repeatedly matches "p q p q p q ... p",
  returning a `std::list<>` of only the values of the p's.  It fails if
  p immediately fails.
* `extension(p)` parses p if strict standard compliance is disabled,
   or with a warning if nonstandard usage warnings are enabled.
* `deprecated(p)` parses p if strict standard compliance is disabled,
  with a warning if deprecated usage warnings are enabled.
* `inContext(msg, p)` runs p within an error message context; any
  message that `p` generates will be tagged with `msg` as its
  context.  Contexts may nest.
* `withMessage(msg, p)` succeeds if `p` does, and if it does not,
  it discards the messages from `p` and fails with the specified message.
* `recovery(p, q)` is equivalent to `p || q`, except that error messages
  generated from the first parser are retained, and a flag is set in
  the ParseState to remember that error recovery was necessary.
* `localRecovery(msg, p, q)` is equivalent to
  `recovery(withMessage(msg, p), q >> pure<A>())` where `A` is the
  result type of 'p'.
  It is useful for targeted error recovery situations within statements.

Note that
```
a >> b >> c / d / e
```
matches a sequence of five parsers, but returns only the result that was
obtained by matching `c`.

### Applicatives
The following *applicative* combinators combine parsers and modify or
collect the values that they return.

* `construct<T>(p1, p2, ...)` matches zero or more parsers in succession,
  collecting their results and then passing them with move semantics to a
  constructor for the type T if they all succeed.
  If there is a single parser as the argument and it returns no usable
  value but only success or failure (_e.g.,_ `"IF"_tok`), the default
  nullary constructor of the type `T` is called.
* `sourced(p)` matches p, and fills in its `source` data member with the
  locations of the cooked character stream that it consumed
* `applyFunction(f, p1, p2, ...)` matches one or more parsers in succession,
  collecting their results and passing them as rvalue reference arguments to
  some function, returning its result.
* `applyLambda([](&&x){}, p1, p2, ...)` is the same thing, but for lambdas
  and other function objects.
* `applyMem(mf, p1, p2, ...)` is the same thing, but invokes a member
  function of the result of the first parser for updates in place.

### Token Parsers
Last, we have these basic parsers on which the actual grammar of the Fortran
is built.  All of the following parsers consume characters acquired from
`nextCh`.

* `space` always succeeds after consuming any spaces
* `spaceCheck` always succeeds after consuming any spaces, and can emit
  a warning if there was no space in free form code before a character
  that could continue a name or keyword
* `digit` matches one cooked decimal digit (0-9)
* `letter` matches one cooked letter (A-Z)
* `"..."_tok` match the content of the string, skipping spaces before and
  after.  Internal spaces are optional matches.  The `_tok` suffix is
  optional when the parser appears before the combinator `>>` or after
  the combinator `/`.  If the quoted string ends in a character that
  could appear in an identifier, a missing space will be diagnosed in
  free form source in pedantic mode if the next character could also
  be part of an identifier -- add a trailing blank to avoid this.
* `"..."_sptok` is a string match in which the spaces are required in
   free form source.
* `"..."_id` is a string match for a complete identifier (not a prefix of
   a longer identifier or keyword).
* `parenthesized(p)` is shorthand for `"(" >> p / ")"`.
* `bracketed(p)` is shorthand for `"[" >> p / "]"`.
* `nonEmptyList(p)` matches a comma-separated list of one or more
  instances of p.
* `nonEmptyList(errorMessage, p)` is equivalent to
  `withMessage(errorMessage, nonemptyList(p))`, which allows one to supply
  a meaningful error message in the event of an empty list.
* `optionalList(p)` is the same thing, but can be empty, and always succeeds.

### Debugging Parser
Last, a string literal `"..."_debug` denotes a parser that emits the string to
`llvm::errs` and succeeds.  It is useful for tracing while debugging a parser but should
obviously not be committed for production code.
