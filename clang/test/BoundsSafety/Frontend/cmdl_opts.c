

// Making sure the Frontend accepts the -fbounds-safety flags - it would otherwise produce error: unknown argument '...'

// RUN: %clang -cc1 -fbounds-safety -fsyntax-only %s

// RUN: %clang -cc1 -fbounds-safety -fexperimental-bounds-safety-cxx -fsyntax-only %s

// RUN: %clang -cc1 -fbounds-safety -fexperimental-bounds-safety-objc -fsyntax-only %s
