// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -Werror -cpp-precomp -fsyntax-only %s

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -Werror -no-cpp-precomp -fsyntax-only %s
