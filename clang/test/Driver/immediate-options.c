// RUN: %clang --help | grep isystem
// RUN: %clang --help | not grep ast-dump
// RUN: %clang --help | not grep ccc-cxx
// RUN: %clang --help-hidden | grep ccc-cxx
// RUN: %clang -dumpversion
// RUN: %clang -print-search-dirs
