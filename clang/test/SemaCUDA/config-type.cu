// RUN: %clang_cc1 -target-sdk-version=8.0 -fsyntax-only -verify=legacy-launch %s
// RUN: %clang_cc1 -target-sdk-version=9.2 -fsyntax-only -verify=new-launch %s

// legacy-launch-error@+1 {{must have scalar return type}}
void cudaConfigureCall(unsigned gridSize, unsigned blockSize);
// new-launch-error@+1 {{must have scalar return type}}
void __cudaPushCallConfiguration(unsigned gridSize, unsigned blockSize);
