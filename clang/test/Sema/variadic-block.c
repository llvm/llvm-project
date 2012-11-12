// RUN: %clang_cc1 %s -verify -fsyntax-only -fblocks
// expected-no-diagnostics

#include <stdarg.h>

int main(int argc, char *argv[]) {
    
    long (^addthem)(const char *, ...) = ^long (const char *format, ...){
        va_list argp;
        const char *p;
        int i;
        char c;
        double d;
        long result = 0;
        va_start(argp, format);
        for (p = format; *p; p++) switch (*p) {
            case 'i':
                i = va_arg(argp, int);
                result += i;
                break;
            case 'd':
                d = va_arg(argp, double);
                result += (int)d;
                break;
            case 'c':
                c = va_arg(argp, int);
                result += c;
                break;
        }
        return result;
    };
    long testresult = addthem("ii", 10, 20);
    if (testresult != 30) {
        return 1;
    }
    testresult = addthem("idc", 30, 40.0, 'a');
    if (testresult != (70+'a')) {
        return 1;
    }
    return 0;
}

