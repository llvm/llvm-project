LLVM core components:
1. Clang frontend: generates LLVM bitcode
2. libc++ C++ standard library: ISO/IE 14882 standard; after object files/IR, references are resolved by linking with libc++ implementations
3. LLD linker: makes one single .exe/library; static linking: all dependencies included; dynamic linking: dependencies linked at runtime)

https://llvm.org/docs/GettingStarted.html#getting-started-with-the-llvm-system
Directory Layout
llvm/cmake
llvm/examples
llvm/include
llvm/lib
llvm/bindings
llvm/projects
llvm/test
test-suite
llvm/tools
llvm/utils

useful info of internals of LLVM generated from source-code: https://llvm.org/doxygen/index.html 


llvm/cmake

generates system build files

/modules: build config for llvm user-defined options. checks compiler version and linker flags

/platforms: toolchain configuration for Android NDK, iOS systems, and non-Windows hosts to target MSVC (Microsoft Visual C++ Compiler)



llvm/examples

examples for using LLVM for a custom language (lowering, optimization, and code generation)

Kaleidoscope language tutorial: hand-written lexer, parser, AST, as well as codegen support using LLVM both static (ahead of time) and various approaches to Just in time (JIT) compilation

JIT: program is compiler into machine code during runtime, right before it's executed

BuildingAJIT: shows how LLVM's ORC JIT APIs interact with other parts of LLVM. teaches how to recombine them to build a custom JIT that is suited to your use-case




llvm/include

public header files exported from the LLVM library. three main directories:

- llvm/include/llvm : all LLVM-specific header files, and subdirectories for different portions of LLVM: Analysis, CodeGen, Target, Transforms, etc., ...
- llvm/include/llvm/support: generic support libraries provided with LLVM but not necessarily specific to LLVM. For example, some C++ STL utilities and a Command Line option processing library store header files here
- llvm/include/llvm/Config: Header files configured by cmake. They wrap 'standard' UNIX and C header files. Source code can include these header files which automatically take care of the conditional #includes that cmake generates.




llvm/lib

most source files are here. by putting code in libraries, LLVM makes it easy to share code among the tools.

llvm/lib/IR - core LLVM source files that implement core classes like Instruction and BasicBlock

llvm/lib/AsmParser - source code for the LLVM assembly language parser library

llvm/lib/Bitcode - code for reading and writing bitcode

llvm/lib/Analysis - a variety of program analyses, such as Call Graphs, Induction Variable, Natural Loop Identification, etc.

llvm/lib/Transforms - IR-to-IR program transformations, such as Aggressive Dead Code Elimination, Sparse Conditional Constant Propagation, Inlining, Loop Invariant Code Motion, Dead Global Elimination, and many others

llvm/lib/Target - files describing target architectures for code generation. for example, llvm/lib/Target/X86 holds the X86 machine description.

llvm/lib/CodeGen - the major parts of the code generator: Instruction Selector, Instruction Scheduling, and Register Allocation.

llvm/lib/MC/ - the libraries represent and process code at machine code level. handles assembly and object-file emission.

llvm/lib/ExecutionEngine - libraries for directly executing bitcode at runtime in interpreted and JIT-compiled scenarios.

llvm/lib/Support - source code that corresponds to the header files in llvm/include/ADT and llvm/include/Support.



















llvm/bindings






















llvm/projects
llvm/test
test-suite
llvm/tools
llvm/utils










LLVM tools include:

Clang: A C, C++, and Objective-C compiler front end. Clang is known for its fast compilation, expressive diagnostics, and adherence to standards. It's often used as an alternative to GCC.

LLVM-AS / LLVM-Dis: Tools for assembling and disassembling LLVM assembly language. llvm-as assembles LLVM assembly into LLVM bitcode, while llvm-dis disassembles LLVM bitcode into human-readable LLVM assembly.

LLVM-Link / LLVM-Archive: llvm-link links LLVM bitcode files together into a single output file. llvm-ar is an archiver that creates and maintains archives of LLVM bitcode files.

LLVM-Opt: This tool provides various optimization options to manipulate LLVM bitcode, allowing developers to run specific optimization passes or transformations on the code.

LLVM-Dump: Used to print the contents of LLVM bitcode files in human-readable form. It's helpful for examining the structure and contents of LLVM bitcode.

LLVM-NM / LLVM-Symbolizer: llvm-nm lists symbols from object files. llvm-symbolizer translates addresses into source code locations for better error reporting and debugging.

LLVM-MC: A machine code generation utility that assembles and disassembles machine code. It's a low-level tool that directly deals with machine instructions and object file formats.

LLVM-ObjDump: Similar to llvm-dis, but specific for object files. It displays information about object files, including their headers, sections, and assembly code.

LLVM-Profdata / LLVM-Cov: These tools deal with code coverage. llvm-profdata manages profile data files, while llvm-cov displays coverage information based on profile data.

LLVM-Size: Displays the size of sections in an LLVM object file or an archive.

