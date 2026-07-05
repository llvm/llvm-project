// Run lines are sensitive to line numbers and come below the code.
// FIXME: re-enable this when we can serialize more C++ ASTs
class Cls {
public:
    Cls operator +(const Cls &RHS);
};

static void bar() {
    Cls x1, x2, x3;
    Cls x4 = x1 + x2 + x3;
}

Cls Cls::operator +(const Cls &RHS) { while (1) {} }

// RUN: %clang_cc1 -emit-pch %s -o %t.ast

// RUNx: index-test %t.ast -point-at %s:10:17 -print-decls > %t &&
// RUNx: cat %t | count 2 &&
// RUNx: grep ':5:9,' %t &&
// RUNx: grep ':13:10,' %t &&

// Yep, we can show references of '+' plus signs that are overloaded, w00t!
// RUNx: index-test %t.ast -point-at %s:5:15 -print-refs > %t &&
// RUNx: cat %t | count 2 &&
// RUNx: grep ':10:17,' %t &&
// RUNx: grep ':10:22,' %t &&

// RUNx: index-test %t.ast -point-at %s:10:14 | grep 'DeclRefExpr x1'
