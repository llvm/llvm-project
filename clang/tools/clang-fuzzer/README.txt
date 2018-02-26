This directory contains two utilities for fuzzing Clang: clang-fuzzer and
clang-proto-fuzzer.  Both use libFuzzer to generate inputs to clang via
coverage-guided mutation.

The two utilities differ, however, in how they structure inputs to Clang.
clang-fuzzer makes no attempt to generate valid C++ programs and is therefore
primarily useful for stressing the surface layers of Clang (i.e. lexer, parser).
clang-proto-fuzzer uses a protobuf class to describe a subset of the C++
language and then uses libprotobuf-mutator to mutate instantiations of that
class, producing valid C++ programs in the process.  As a result,
clang-proto-fuzzer is better at stressing deeper layers of Clang and LLVM.

===================================
 Building clang-fuzzer
===================================
Within your LLVM build directory, run CMake with the following variable
definitions:
- CMAKE_C_COMPILER=clang
- CMAKE_CXX_COMPILER=clang++
- LLVM_USE_SANITIZE_COVERAGE=YES
- LLVM_USE_SANITIZER=Address

Then build the clang-fuzzer target.

Example:
  cd $LLVM_SOURCE_DIR
  mkdir build && cd build
  cmake .. -GNinja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_SANITIZE_COVERAGE=YES -DLLVM_USE_SANITIZER=Address
  ninja clang-fuzzer

======================
 Running clang-fuzzer
======================
  bin/clang-fuzzer CORPUS_DIR


=======================================================
 Building clang-proto-fuzzer (Linux-only instructions)
=======================================================
Install the necessary dependencies:
- binutils  // needed for libprotobuf-mutator
- liblzma-dev  // needed for libprotobuf-mutator
- libz-dev  // needed for libprotobuf-mutator
- docbook2x  // needed for libprotobuf-mutator
- Recent version of protobuf [3.3.0 is known to work]

Within your LLVM build directory, run CMake with the following variable
definitions:
- CMAKE_C_COMPILER=clang
- CMAKE_CXX_COMPILER=clang++
- LLVM_USE_SANITIZE_COVERAGE=YES
- LLVM_USE_SANITIZER=Address
- CLANG_ENABLE_PROTO_FUZZER=ON

Then build the clang-proto-fuzzer and clang-proto-to-cxx targets.  Optionally,
you may also build clang-fuzzer with this setup.

Example:
  cd $LLVM_SOURCE_DIR
  mkdir build && cd build
  cmake .. -GNinja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_SANITIZE_COVERAGE=YES -DLLVM_USE_SANITIZER=Address \
    -DCLANG_ENABLE_PROTO_FUZZER=ON
  ninja clang-proto-fuzzer clang-proto-to-cxx

This directory also contains a Dockerfile which sets up all required
dependencies and builds the fuzzers.

============================
 Running clang-proto-fuzzer
============================
  bin/clang-proto-fuzzer CORPUS_DIR

Arguments can be specified after -ignore_remaining_args=1 to modify the compiler
invocation.  For example, the following command line will fuzz LLVM with a
custom optimization level and target triple:
  bin/clang-proto-fuzzer CORPUS_DIR -ignore_remaining_args=1 -O3 -triple \
      arm64apple-ios9

To translate a clang-proto-fuzzer corpus output to C++:
  bin/clang-proto-to-cxx CORPUS_OUTPUT_FILE
