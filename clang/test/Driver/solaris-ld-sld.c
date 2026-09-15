// REQUIRES: system-solaris

// Check that clang invokes the native ld, not whatever "ld" that happens to be in PATH

// RUN: env PATH=%S/Inputs/fake_ld/ %clang -v -o %t.o %s

int main() { return 0; }
