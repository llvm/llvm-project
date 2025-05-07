
// Making sure the Frontend accepts the -fbounds-safety flags - it would otherwise produce error: unknown argument '...'

// RUN: %clang -fbounds-safety -fsyntax-only %s

// RUN: %clang -fbounds-safety -Xclang -fexperimental-bounds-safety-cxx -fsyntax-only %s

// RUN: %clang -fbounds-safety -Xclang -fexperimental-bounds-safety-objc -fsyntax-only %s
