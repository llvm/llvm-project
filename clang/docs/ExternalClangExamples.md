# External Clang Examples

## Introduction

This page provides some examples of the kinds of things that people have
done with Clang that might serve as useful guides (or starting points) from
which to develop your own tools. They may be helpful even for something as
banal (but necessary) as how to set up your build to integrate Clang.

Clang's library-based design is deliberately aimed at facilitating use by
external projects, and we are always interested in improving Clang to
better serve our external users. Some typical categories of applications
where Clang is used are:

- Static analysis.
- Documentation/cross-reference generation.

If you know of (or wrote!) a tool or project using Clang, please post on
[the Discourse forums (Clang Frontend category)](https://discourse.llvm.org/c/clang/6) to have it added.
(or if you are already a Clang contributor, feel free to directly commit
additions). Since the primary purpose of this page is to provide examples
that can help developers, generally they must have code available.

## List of projects and tools

[https://github.com/Andersbakken/rtags/](https://github.com/Andersbakken/rtags/)

: "RTags is a client/server application that indexes c/c++ code and keeps
  a persistent in-memory database of references, symbolnames, completions
  etc."

[https://rprichard.github.io/CxxCodeBrowser/](https://rprichard.github.io/CxxCodeBrowser/)

: "A C/C++ source code indexer and navigator."

[https://github.com/etaoins/qconnectlint](https://github.com/etaoins/qconnectlint)

: "qconnectlint is a Clang tool for statically verifying the consistency
  of signal and slot connections made with Qt's `QObject::connect`."

[https://github.com/woboq/woboq_codebrowser](https://github.com/woboq/woboq_codebrowser)

: "The Woboq Code Browser is a web-based code browser for C/C++ projects.
  Check out [https://code.woboq.org/](https://code.woboq.org/) for an example!"

[https://github.com/mozilla/dxr](https://github.com/mozilla/dxr)

: "DXR is a source code cross-reference tool that uses static analysis
  data collected by instrumented compilers."

[https://github.com/eschulte/clang-mutate](https://github.com/eschulte/clang-mutate)

: "This tool performs a number of operations on C-language source files."

[https://github.com/gmarpons/Crisp](https://github.com/gmarpons/Crisp)

: "A coding rule validation add-on for LLVM/clang. Crisp rules are written
  in Prolog. A high-level declarative DSL to easily write new rules is under
  development. It will be called CRISP, an acronym for *Coding Rules in
  Sugared Prolog*."

[https://github.com/drothlis/clang-ctags](https://github.com/drothlis/clang-ctags)

: "Generate tag file for C++ source code."

[https://github.com/exclipy/clang_indexer](https://github.com/exclipy/clang_indexer)

: "This is an indexer for C and C++ based on the libclang library."

[https://github.com/holtgrewe/linty](https://github.com/holtgrewe/linty)

: "Linty - C/C++ Style Checking with Python & libclang."

[https://github.com/axw/cmonster](https://github.com/axw/cmonster)

: "cmonster is a Python wrapper for the Clang C++ parser."

[https://github.com/rizsotto/Constantine](https://github.com/rizsotto/Constantine)

: "Constantine is a toy project to learn how to write clang plugin.
  Implements pseudo const analysis. Generates warnings about variables,
  which were declared without const qualifier."

[https://github.com/jessevdk/cldoc](https://github.com/jessevdk/cldoc)

: "cldoc is a Clang based documentation generator for C and C++.
  cldoc tries to solve the issue of writing C/C++ software documentation
  with a modern, non-intrusive and robust approach."

[https://github.com/AlexDenisov/ToyClangPlugin](https://github.com/AlexDenisov/ToyClangPlugin)

: "The simplest Clang plugin implementing a semantic check for Objective-C.
  This example shows how to use the `DiagnosticsEngine` (emit warnings,
  errors, fixit hints). See also [http://l.rw.rw/clang_plugin](http://l.rw.rw/clang_plugin) for
  step-by-step instructions."

[https://phabricator.kde.org/source/clazy](https://phabricator.kde.org/source/clazy)

: "clazy is a compiler plugin which allows clang to understand Qt semantics.
  You get more than 50 Qt related compiler warnings, ranging from unneeded
  memory allocations to misusage of API, including fix-its for automatic
  refactoring."

[https://gerrit.libreoffice.org/gitweb?p=core.git;a=blob_plain;f=compilerplugins/README;hb=HEAD](https://gerrit.libreoffice.org/gitweb?p=core.git;a=blob_plain;f=compilerplugins/README;hb=HEAD)

: "LibreOffice uses a Clang plugin infrastructure to check during the build
  various things, some more, some less specific to the LibreOffice source code.
  There are currently around 50 such checkers, from flagging C-style casts and
  uses of reserved identifiers to ensuring that code adheres to lifecycle
  protocols for certain LibreOffice-specific classes. They may serve as
  examples for writing RecursiveASTVisitor-based plugins."

[https://github.com/banach-space/clang-tutor](https://github.com/banach-space/clang-tutor)

: "A collection of out-of-tree Clang plugins for teaching and learning."

