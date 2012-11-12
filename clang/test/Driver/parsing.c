// RUN: %clang -ccc-print-options input -Yunknown -m32 -arch ppc -djoined -A separate -Ajoined -Wp,one,two -Xarch_joined AndSeparate -sectalign 1 2 3 2> %t
// RUN: grep 'Option 0 - Name: "-ccc-print-options", Values: {}' %t
// RUN: grep 'Option 1 - Name: "<input>", Values: {"input"}' %t
// RUN: grep 'Option 2 - Name: "<unknown>", Values: {"-Yunknown"}' %t
// RUN: grep 'Option 3 - Name: "-m32", Values: {}' %t
// RUN: grep 'Option 4 - Name: "-arch", Values: {"ppc"}' %t
// RUN: grep 'Option 5 - Name: "-d", Values: {"joined"}' %t
// RUN: grep 'Option 6 - Name: "-A", Values: {"separate"}' %t
// RUN: grep 'Option 7 - Name: "-A", Values: {"joined"}' %t
// RUN: grep 'Option 8 - Name: "-Wp,", Values: {"one", "two"}' %t
// RUN: grep 'Option 9 - Name: "-Xarch_", Values: {"joined", "AndSeparate"}' %t
// RUN: grep 'Option 10 - Name: "-sectalign", Values: {"1", "2", "3"}' %t

// RUN: not %clang -V 2> %t
// RUN: grep "error: argument to '-V' is missing (expected 1 value)" %t
// RUN: not %clang -sectalign 1 2 2> %t
// RUN: grep "error: argument to '-sectalign' is missing (expected 3 values)" %t

// Verify that search continues after find the first option.
// RUN: %clang -ccc-print-options -Wally 2> %t
// RUN: grep 'Option 0 - Name: "-ccc-print-options", Values: {}' %t
// RUN: grep 'Option 1 - Name: "-W", Values: {"ally"}' %t
