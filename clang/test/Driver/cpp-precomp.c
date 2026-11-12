// RUN: %clang -target x86_64-apple-darwin11 \
// RUN:   -Werror -cpp-precomp -fsyntax-only %s

// RUN: %clang -target x86_64-apple-darwin11 \
// RUN:   -Werror -no-cpp-precomp -fsyntax-only %s
