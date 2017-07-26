// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -verify %s

/* expected-warning {{value of #pragma pack(show) == 8}} */ #pragma pack(show)
/* expected-warning {{expected #pragma pack parameter to be}} */ #pragma pack(3)
/* expected-warning {{value of #pragma pack(show) == 8}} */ #pragma pack(show)
#pragma pack(4)
/* expected-warning {{value of #pragma pack(show) == 4}} */ #pragma pack(show)
#pragma pack() // resets to default
/* expected-warning {{value of #pragma pack(show) == 8}} */ #pragma pack(show)
#pragma pack(2)
#pragma pack(push, eek, 16) // -> (eek, 2), 16
/* expected-warning {{value of #pragma pack(show) == 16}} */ #pragma pack(show)
#pragma pack(push) // -> (eek, 2), (, 2), 16
/* expected-warning {{value of #pragma pack(show) == 16}} */ #pragma pack(show)
#pragma pack(1) 
#pragma pack(push, 8) // -> (eek, 2), (, 2), (, 1), 8
/* expected-warning {{value of #pragma pack(show) == 8}} */ #pragma pack(show)
#pragma pack(pop) // -> (eek, 2), (,2), 1
/* expected-warning {{value of #pragma pack(show) == 1}} */ #pragma pack(show)
#pragma pack(pop, eek)
/* expected-warning {{value of #pragma pack(show) == 2}} */ #pragma pack(show)
/* expected-warning {{pack(pop, ...) failed: stack empty}} */ #pragma pack(pop)

#pragma pack(push)
#pragma pack(pop, 16)
/* expected-warning {{value of #pragma pack(show) == 16}} */ #pragma pack(show)

