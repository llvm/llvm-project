// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -fblocks -triple x86_64-apple-darwin10.0.0 %s
// expected-no-diagnostics

bool fn(id obj) {
    return (bool)obj;
}

