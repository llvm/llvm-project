// UNSUPPORTED: system-windows
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/level1/level2

// RUN: cd %t.dir
// RUN: echo "*" > .clang-format-ignore
// RUN: echo "level*/*.c*" >> .clang-format-ignore
// RUN: echo "*/*2/foo.*" >> .clang-format-ignore
// RUN: touch foo.cc
// RUN: clang-format -verbose .clang-format-ignore foo.cc 2> %t.stderr
// RUN: not grep Formatting %t.stderr

// RUN: cd level1
// RUN: touch bar.cc baz.c
// RUN: clang-format -verbose bar.cc baz.c 2> %t.stderr
// RUN: not grep Formatting %t.stderr

// RUN: cd level2
// RUN: touch foo.c foo.js
// RUN: clang-format -verbose foo.c foo.js 2> %t.stderr
// RUN: not grep Formatting %t.stderr

// RUN: touch .clang-format-ignore
// RUN: clang-format -verbose foo.c foo.js 2> %t.stderr
// RUN: grep -Fx "Formatting [1/2] foo.c" %t.stderr
// RUN: grep -Fx "Formatting [2/2] foo.js" %t.stderr

// RUN: echo "*.js" > .clang-format-ignore
// RUN: clang-format -verbose foo.c foo.js 2> %t.stderr
// RUN: grep -Fx "Formatting [1/2] foo.c" %t.stderr
// RUN: not grep -F foo.js %t.stderr

// RUN: cd ../..
// RUN: clang-format -verbose *.cc level1/*.c* level1/level2/foo.* 2> %t.stderr
// RUN: grep -x "Formatting \[1/5] .*foo\.c" %t.stderr
// RUN: not grep -F foo.js %t.stderr

// RUN: rm .clang-format-ignore
// RUN: clang-format -verbose *.cc level1/*.c* level1/level2/foo.* 2> %t.stderr
// RUN: grep -x "Formatting \[1/5] .*foo\.cc" %t.stderr
// RUN: grep -x "Formatting \[2/5] .*bar\.cc" %t.stderr
// RUN: grep -x "Formatting \[3/5] .*baz\.c" %t.stderr
// RUN: grep -x "Formatting \[4/5] .*foo\.c" %t.stderr
// RUN: not grep -F foo.js %t.stderr

// RUN: cd ..
// RUN: rm -r %t.dir
