// Ensure we support the -mtune flag.
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -mtune=nocona 2>&1 \
// RUN:   | FileCheck %s -check-prefix=nocona
// nocona: "-tune-cpu" "nocona"
