// RUN: %clang -### \
// RUN:   -M -MM %s 2> %t
// RUN: grep '"-sys-header-deps"' %t | count 0
