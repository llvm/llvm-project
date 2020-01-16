// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

struct AAA
{
    struct BBB
    {
        ~BBB() {}
    };

    typedef BBB BBB_alias;
};

typedef AAA::BBB BBB_alias2;

int
main()
{
    AAA::BBB_alias *ptr1 = new AAA::BBB_alias();
    AAA::BBB_alias *ptr2 = new AAA::BBB_alias();

    ptr1->AAA::BBB_alias::~BBB_alias(); // Now OK
    ptr2->AAA::BBB_alias::~BBB();       // OK
    ptr1->~BBB_alias2();                // OK
    return 0;
}
