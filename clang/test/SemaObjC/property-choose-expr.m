// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface NSArray
-(int)count;
@end

// <rdar://problem/14438917>
char* f(NSArray *array) {
    return _Generic(__builtin_choose_expr(__builtin_types_compatible_p(__typeof__(array.count), void), 0.f, array.count),
                    unsigned int:"uint",
                    float:"void",
                    default: "ignored");
}
