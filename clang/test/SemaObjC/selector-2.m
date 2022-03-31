// RUN: %clang_cc1  -fsyntax-only -Wselector -verify %s
// expected-no-diagnostics
// rdar://8851684
@interface  I
- length;
@end

static inline SEL IsEmpty(void) {
    return @selector(length);
}

int main (int argc, const char * argv[]) {
    return 0;
}

