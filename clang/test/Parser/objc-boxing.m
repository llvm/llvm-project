// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSString @end

@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

extern char *strdup(const char *str);

id constant_string(void) {
    return @("boxed constant string.");
}

id dynamic_string(void) {
    return @(strdup("boxed dynamic string"));
}

id const_char_pointer(void) {
    return @((const char *)"constant character pointer");
}

id missing_parentheses(void) {
    return @(5;             // expected-error {{expected ')'}} \
                            // expected-note {{to match this '('}}
}

void bar(id p);
void foo(id p) {
        bar(@{p, p}); // expected-error {{expected ':'}}
        bar(0);
        bar(0);
}
