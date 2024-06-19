// RUN: %clangxx -std=c++11 -fsyntax-only -Xclang -verify %s 
#include <iostream>
using namespace std;
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

int main() {
    // Check valid usage
    cout<<"Symbolic Name for "<<Day::Tuesday<<" is: "<<__nameof(Day::Tuesday)<<endl; // expected-output {{Symbolic Name for 1 is: Day::Tuesday}}
    cout<<"Symbolic Name for "<<Month::August<<" is: "<<__nameof(Month::August)<<endl; // expected-output {{Symbolic Name for 7 is: Month::August}}
    // Check invalid usage
    cout<<"Symbolic Name: "<<__nameof(1)<<endl;  // expected-error {{unsupported declaration type. Only enum constants are supported}}
    cout<<"Symbolic Name: "<<__nameof("")<<endl;  // expected-error {{unsupported declaration type. Only enum constants are supported}}
    return 0;
}