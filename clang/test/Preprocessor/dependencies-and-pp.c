// Test -MT and -E flags, PR4063

// RUN: %clang -E -o %t.1 %s
// RUN: %clang -E -MD -MF %t.d -MT foo -o %t.2 %s
// RUN: diff %t.1 %t.2
// RUN: grep "foo:" %t.d
// RUN: grep "dependencies-and-pp.c" %t.d

// Test -MQ flag without quoting

// RUN: %clang -E -MD -MF %t.d -MQ foo -o %t %s
// RUN: grep "foo:" %t.d

// Test -MQ flag with quoting

// RUN: %clang -E -MD -MF %t.d -MQ '$fo\ooo ooo\ ooo\\ ooo#oo' -o %t %s
// RUN: fgrep '$$fo\ooo\ ooo\\\ ooo\\\\\ ooo\#oo:' %t.d

// Test consecutive -MT flags

// RUN: %clang -E -MD -MF %t.d -MT foo -MT bar -MT baz -o %t %s
// RUN: diff %t.1 %t
// RUN: fgrep "foo bar baz:" %t.d

// Test consecutive -MT and -MQ flags

// RUN: %clang -E -MD -MF %t.d -MT foo -MQ '$(bar)' -MT 'b az' -MQ 'qu ux' -MQ ' space' -o %t %s
// RUN: fgrep 'foo $$(bar) b az qu\ ux \ space:' %t.d

// TODO: Test default target without quoting
// TODO: Test default target with quoting
