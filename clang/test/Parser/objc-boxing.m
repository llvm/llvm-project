// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface NSString @end

@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

extern char *strdup(const char *str);

id constant_string() {
    return @("boxed constant string.");
}

id dynamic_string() {
    return @(strdup("boxed dynamic string"));
}

id const_char_pointer() {
    return @((const char *)"constant character pointer");
}

id missing_parentheses() {
    return @(5;             // expected-error {{expected ')'}} \
                            // expected-note {{to match this '('}}
}
