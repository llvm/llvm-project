

// Making sure the Frontend accepts the -fbounds-safety flags - it would otherwise produce error: unknown argument '...'

// RUN: %clang -cc1 -fbounds-safety -fsyntax-only %s

// RUN: %clang -cc1 -fbounds-safety -fbounds-attributes-cxx-experimental -fsyntax-only %s

// RUN: %clang -cc1 -fbounds-safety -fbounds-attributes-objc-experimental -fsyntax-only %s
