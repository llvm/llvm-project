# Adding Programming Language Support

LLDB has been architected to make it straightforward to add support for a
programming language. Only a small enum in core LLDB needs to be modified to
make LLDB aware of a new programming language. Everything else can be supplied
in derived classes that need not even be present in the core LLDB repository.
This makes it convenient for developers adding language support in downstream
repositories since it practically eliminates the potential for merge conflicts.

The basic steps are:
* Add the language to the `LanguageType` enum.
* Add a `TypeSystem` for the language.
* Add expression evaluation support.

Additionally, you may want to create a `Language` and `LanguageRuntime` plugin
for your language, which enables support for advanced features like dynamic
typing and data formatting.

## Add the Language to the LanguageType enum

The `LanguageType` enum
(see [lldb-enumerations.h](https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/lldb-enumerations.h))
contains a list of every language known to LLDB. It is the one place where
support for a language must live that will need to merge cleanly with upstream
LLDB if you are developing your language support in a separate branch. When
adding support for a language previously unknown to LLDB, start by adding an
enumeration entry to `LanguageType`.

## Add a TypeSystem for the Language

Both [Module](https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/Core/Module.h)
and [Target](https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/Target/Target.h)
support the retrieval of a `TypeSystem` instance via `GetTypeSystemForLanguage()`.
For `Module`, this method is directly on the `Module` instance. For `Target`,
this is retrieved indirectly via the `TypeSystemMap` for the `Target` instance.

The `TypeSystem` instance returned by the `Target` is expected to be capable of
evaluating expressions, while the `TypeSystem` instance returned by the `Module`
is not. If you want to support expression evaluation for your language, you could
consider one of the following approaches:
* Implement a single `TypeSystem` class that supports evaluation when given an
  optional `Target`, implementing all the expression evaluation methods on the
  `TypeSystem`.
* Create multiple `TypeSystem` classes, one for evaluation and one for static
  `Module` usage.

For clang and Swift, the latter approach was chosen. Primarily to make it
clearer that evaluation with the static `Module`-returned `TypeSystem` instances
make no sense, and have them error out on those calls. But either approach is
fine.

# Creating Types

Your `TypeSystem` will need an approach for creating types based on a set of
`Module`s. If your type info is going to come from DWARF info, you will want to
subclass [DWARFASTParser](https://github.com/llvm/llvm-project/blob/main/lldb/source/Plugins/SymbolFile/DWARF/DWARFASTParser.h).


# Add Expression Evaluation Support

Expression Evaluation support is enabled by implementing the relevant methods on
a `TypeSystem`-derived class. Search for `Expression` in the
[TypeSystem header](https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/Symbol/TypeSystem.h)
to find the methods to implement.

# Type Completion

There are three levels of type completion, each requiring more type information:
1. Pointer size: When you have a forward decl or a reference, and that's all you
  need. At this stage, the pointer size is all you need.
2. Layout info: You need the size of an instance of the type, but you still don't
  need to know all the guts of the type.
3. Full type info: Here you need everything, because you're playing with
  internals of it, such as modifying a member variable.

Ensure you never complete more of a type than is needed for a given situation.
This will keep your type system from doing more work than necessary.

# Language and LanguageRuntime Plugins

If you followed the steps outlined above, you already have taught LLDB a great
deal about your language. If your language's runtime model and fundamental data
types don't differ much from the C model, you are pretty much done.

However it is likely that your language offers its own data types for things
like strings and arrays, and probably has a notion of dynamic types, where the
effective type of a variable can only be known at runtime.

These tasks are covered by two plugins:
* a `LanguageRuntime` plugin, which provides LLDB with a dynamic view of your
  language; this plugin answers questions that require a live process to acquire
  information (for example dynamic type resolution).
* a `Language` plugin, which provides LLDB with a static view of your language;
  questions that are statically knowable and do not require a process are
  answered by this plugin (for example data formatters).