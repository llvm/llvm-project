// RUN: %clang_cc1 -pedantic -fixit -x c %s
// rdar://9091893

#define va_arg(ap, type)    __builtin_va_arg(ap, type)
typedef __builtin_va_list va_list;

void myFunc(void) {
    va_list values;
    
    int value;

    while (value = va_arg(values, int)) {  // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                                           // expected-note {{use '==' to turn this assignment into an equality comparison}} \
					   // expected-note {{place parentheses around the assignment to silence this warning}}
    }
}
