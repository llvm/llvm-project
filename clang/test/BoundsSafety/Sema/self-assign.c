
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wself-assign -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wself-assign -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// -Wself-assign should not trigger for self-assignments of variables that have
// external bounds, or on the external bounds of those variables.

void count(int *__counted_by(count) p, int count) {
    p = p;
    count = count;
    if (1) {
        int count2 = count;
        int *__counted_by(count2) p2 = p;
        count2 = count2;
        p2 = p2;
    }
}

void end(int *__ended_by(end) begin, int *end) {
    begin = begin;
    end = end;
    if (1) {
        int *end2 = end;
        int *__ended_by(end2) begin2 = begin;
        begin2 = begin2;
        end2 = end2;
    }
}

void unrelated(int unrelated) {
    unrelated = unrelated; // expected-warning{{explicitly assigning value of variable of type 'int' to itself}}
}
