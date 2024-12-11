
// Making sure the Frontend accepts the -fbounds-safety flags in assembly mode - it would otherwise produce error: unknown argument '...'

// RUN: %clang -fsyntax-only -fbounds-safety %s
