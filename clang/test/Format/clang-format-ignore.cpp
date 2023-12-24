// RUN: mkdir -p %t.dir/level1/level2

// RUN: cd %t.dir
// RUN: printf "*\nlevel*/*.c*\n*/*2/foo.*\n" > .clang-format-ignore
// RUN: touch foo.cc
// RUN: clang-format -verbose .clang-format-ignore foo.cc 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr

// RUN: cd level1
// RUN: touch bar.cc baz.c
// RUN: clang-format -verbose bar.cc baz.c 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr

// RUN: cd level2
// RUN: touch foo.c foo.js
// RUN: clang-format -verbose foo.c foo.js 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr
// RUN: printf "*.js\n" > .clang-format-ignore
// RUN: clang-format -verbose foo.c foo.js 2> %t.stderr
// RUN: grep -E "Formatting (.*)foo.c(.*)" %t.stderr
// RUN: not grep -E "Formatting (.*)foo.js(.*)" %t.stderr

// RUN: cd ../../..
// RUN: rm -rf %t.dir
