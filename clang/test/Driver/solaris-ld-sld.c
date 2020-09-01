// REQUIRES: system-solaris

// Check that clang invokes the native ld.

// RUN: test -f /usr/gnu/bin/ld && env PATH=/usr/gnu/bin %clang -o %t.o %s

int main() { return 0; }
