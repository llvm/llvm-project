// RUN: %clangxx -std=c++11 -fsyntax-only -Xclang -verify %s

int printf(const char *restrict, ...);

// Define an enumeration
enum Day {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday
};

enum Month {
    January,
    February,
    March,
    April,
    May,
    June,
    July,
    August,
    September,
    October,
    November,
    December
};

void test_enum_nameof() {
    // Check valid usage
    printf("Symbolic Name for %d is: %s ", Day::Tuesday, __nameof(Day::Tuesday)); // expected-output {{Symbolic Name for 1 is: Day::Tuesday}}
    printf("Symbolic Name for %d is: %s ", Month::August, __nameof(Month::August)); // expected-output {{Symbolic Name for 7 is: Month::August}}
    
    // Check invalid usage
    printf("Symbolic Name: %s ", __nameof(1));  // expected-error {{unsupported declaration type. Only enum constants are supported}}
    printf("Symbolic Name: %s ", __nameof(""));  // expected-error {{unsupported declaration type. Only enum constants are supported}}
}