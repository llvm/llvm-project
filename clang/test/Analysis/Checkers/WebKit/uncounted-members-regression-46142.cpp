// regression test for https://bugs.llvm.org/show_bug.cgi?id=46142

// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.WebKitNoUncountedMemberChecker -verify %s
// expected-no-diagnostics

class ClassWithoutADefinition;
class Foo {
    const ClassWithoutADefinition *foo;
};
