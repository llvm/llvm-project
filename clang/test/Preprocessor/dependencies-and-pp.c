// Test -MT and -E flags, PR4063

// RUN: %clang -E -o %t.1 %s
// RUN: %clang -E -MD -MF %t.d -MT foo -o %t.2 %s
// RUN: diff %t.1 %t.2
// RUN: FileCheck -check-prefix=TEST1 %s < %t.d
// TEST1: foo:
// TEST1: dependencies-and-pp.c

// Test -MQ flag without quoting

// RUN: %clang -E -MD -MF %t.d -MQ foo -o %t %s
// RUN: FileCheck -check-prefix=TEST2 %s < %t.d
// TEST2: foo:

// Test -MQ flag with quoting

// RUN: %clang -E -MD -MF %t.d -MQ '$fo\ooo ooo\ ooo\\ ooo#oo' -o %t %s
// RUN: FileCheck -check-prefix=TEST3 %s < %t.d
// TEST3: $$fo\ooo\ ooo\\\ ooo\\\\\ ooo\#oo:

// Test consecutive -MT flags

// RUN: %clang -E -MD -MF %t.d -MT foo -MT bar -MT baz -o %t %s
// RUN: diff %t.1 %t
// RUN: FileCheck -check-prefix=TEST4 %s < %t.d
// TEST4: foo bar baz:

// Test consecutive -MT and -MQ flags

// RUN: %clang -E -MD -MF %t.d -MT foo -MQ '$(bar)' -MT 'b az' -MQ 'qu ux' -MQ ' space' -o %t %s
// RUN: FileCheck -check-prefix=TEST5 %s < %t.d
// TEST5: foo $$(bar) b az qu\ ux \ space:

// Test self dependency, PR31644

// RUN: %clang -E -MD -MP -MF %t.d %s
// RUN: FileCheck -check-prefix=TEST6 %s < %t.d
// TEST6: dependencies-and-pp.c
// TEST6-NOT: dependencies-and-pp.c:

// TODO: Test default target without quoting
// TODO: Test default target with quoting
