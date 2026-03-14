// Check that on the PS4 we default to:
// -target-cpu btver2, not -tune-cpu generic
// And on the PS5 we default to:
// -target-cpu znver2, not -tune-cpu generic

// RUN: %clang -target x86_64-scei-ps4 -c %s -### 2>&1 | FileCheck --check-prefixes=PS4,BOTH %s
// RUN: %clang -target x86_64-sie-ps5 -c %s -### 2>&1 | FileCheck --check-prefixes=PS5,BOTH %s
// PS4: "-target-cpu" "btver2"
// PS5: "-target-cpu" "znver2"
// BOTH-NOT: "-tune-cpu"
