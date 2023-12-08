// Test using -x cuda -fopenmp does not clash integrated headers.
// Reported in https://bugs.llvm.org/show_bug.cgi?id=48014
///==========================================================================///

// REQUIRES: nvptx-registered-target

// RUN: %clang -x cuda -fopenmp -c %s -o - --cuda-path=%S/../Driver/Inputs/CUDA/usr/local/cuda -nocudalib -isystem %S/Inputs/include -isystem %S/../../lib/Headers -fsyntax-only

