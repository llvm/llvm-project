Goals
=====

The current state of the art in open source debuggers are that they work in the
common cases for C applications, but don't handle many "hard cases" properly.
For example, C++ expression parsing, handling overloading, templates,
multi-threading, and other non-trivial scenarios all work in some base cases,
but don't work reliably.

The goal of LLDB is to provide an amazing debugging experience that "just
works". We aim to solve these long-standing problems where debuggers get
confused, so that you can think about debugging your problem, not about
deficiencies in the debugger.

With a long view, there is no good reason for a debugger to reinvent its own
C/C++ parser, type system, know all the target calling convention details,
implement its own disassembler, etc. By using the existing libraries vended by
the LLVM project, we believe that many of these problems will be defined away,
and the debugger can focus on important issues like process control, efficient
symbol reading and indexing, thread management, and other debugger-specific
problems.

Some more specific goals include:

* Build libraries for inclusion in IDEs, command line tools, and other analysis
  tools
* High performance and efficient memory use
* Extensible: Python scriptable and use a plug-in architecture
* Reuse existing compiler technology where it makes sense
* Excellent multi-threaded debugging support
* Great support for C, Objective-C and C++
* Retargetable to support multiple platforms
* Provide a base for debugger research and other innovation
