// RUN: %clang -E %s "-DX=A
// RUN: THIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT" > %t
// RUN: grep "GOOD: A" %t
// RUN: not grep THIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT %t
// rdar://6762183

GOOD: X

