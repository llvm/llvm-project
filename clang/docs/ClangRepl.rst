===========
Clang-Repl
===========

**Clang-Repl** is an interactive C++ interpreter that allows for incremental
compilation. It supports interactive programming for C++ in a
read-evaluate-print-loop (REPL) style. It uses Clang as a library to compile the
high level programming language into LLVM IR. Then the LLVM IR is executed by
the LLVM just-in-time (JIT) infrastructure.

Clang-Repl is suitable for exploratory programming and in places where time
to insight is important. Clang-Repl is a project inspired by the work in
`Cling <https://github.com/root-project/cling>`_, a LLVM-based C/C++ interpreter
developed by the field of high energy physics and used by the scientific data
analysis framework `ROOT <https://root.cern/>`_. Clang-Repl allows to move parts
of Cling upstream, making them useful and available to a broader audience.



Clang-Repl Usage
================


.. code-block:: text

  clang-repl> #include <iostream>
  clang-repl> int f() { std::cout << "Hello Interpreted World!\n"; return 0; }
  clang-repl> auto r = f();
  // Prints Hello Interpreted World!

Note that the implementation is not complete and highly experimental. We do
not yet support statements on the global scope, for example.


Clang-Repl Basic Data Flow
==========================

.. image:: ClangRepl_design.png
   :align: center
   :alt: ClangRepl design

Clang-Repl data flow can be divided into roughly 8 phases:

1. Clang-Repl controls the input infrastructure by an interactive prompt or by
   an interface allowing the incremental processing of input.

2. Then it sends the input to the underlying incremental facilities in Clang
   infrastructure.

3. Clang compiles the input into an AST representation.

4. When required the AST can be further transformed in order to attach specific
   behavior.

5. The AST representation is then lowered to LLVM IR.

6. The LLVM IR is the input format for LLVM’s JIT compilation infrastructure.
   The tool will instruct the JIT to run specified functions, translating them
   into machine code targeting the underlying device architecture (eg. Intel
   x86 or NVPTX).

7. The LLVM JIT lowers the LLVM IR to machine code.

8. The machine code is then executed.


Just like Clang, Clang-Repl can be integrated in existing applications as a
library (via using the clangInterpreter library). This turning your C++ compiler
into a service which incrementally can consume and execute code. The
**Compiler as A Service** (**CaaS**) concept helps supporting move advanced use
cases such as template instantiations on demand and automatic language
interoperability. It also helps static languages such as C/C++ become apt for
data science.


Related Reading
===============
`Cling Transitions to LLVM's Clang-Repl <https://root.cern/blog/cling-in-llvm/>`_

`Moving (parts of) the Cling REPL in Clang <https://lists.llvm.org/pipermail/llvm-dev/2020-July/143257.html>`_

`GPU Accelerated Automatic Differentiation With Clad <https://arxiv.org/pdf/2203.06139.pdf>`_
