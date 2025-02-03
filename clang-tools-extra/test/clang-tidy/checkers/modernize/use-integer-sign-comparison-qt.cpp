// CHECK-FIXES: #include <QtCore/q20utility.h>
// RUN: %check_clang_tidy -std=c++17 %s modernize-use-integer-sign-comparison %t -- \
// RUN: -config="{CheckOptions: {modernize-use-integer-sign-comparison.EnableQtSupport: true}}"

// The code that triggers the check
#define MAX_MACRO(a, b) (a < b) ? b : a

unsigned int FuncParameters(int bla) {
    unsigned int result = 0;
    if (result == bla)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_equal(result , bla))

    return 1;
}

template <typename T>
void TemplateFuncParameter(T val) {
    unsigned long uL = 0;
    if (val >= uL)
        return;
// CHECK-MESSAGES-NOT: warning:
}

template <typename T1, typename T2>
int TemplateFuncParameters(T1 val1, T2 val2) {
    if (val1 >= val2)
        return 0;
// CHECK-MESSAGES-NOT: warning:
    return 1;
}

int AllComparisons() {
    unsigned int uVar = 42;
    unsigned short uArray[7] = {0, 1, 2, 3, 9, 7, 9};

    int sVar = -42;
    short sArray[7] = {-1, -2, -8, -94, -5, -4, -6};

    enum INT_TEST {
      VAL1 = 0,
      VAL2 = -1
    };

    char ch = 'a';
    unsigned char uCh = 'a';
    signed char sCh = 'a';
    bool bln = false;

    if (bln == sVar)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (ch > uCh)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (sVar <= INT_TEST::VAL2)
      return 0;
// CHECK-MESSAGES-NOT: warning:

    if (uCh < sCh)
      return -1;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_less(uCh , sCh))

    if ((int)uVar < sVar)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_less(uVar, sVar))

    (uVar != sVar) ? uVar = sVar
                   : sVar = uVar;
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: (q20::cmp_not_equal(uVar , sVar)) ? uVar = sVar

    while (uArray[0] <= sArray[0])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: while (q20::cmp_less_equal(uArray[0] , sArray[0]))

    if (uArray[1] > sArray[1])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_greater(uArray[1] , sArray[1]))

    MAX_MACRO(uVar, sArray[0]);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]

    if (static_cast<unsigned int>(uArray[2]) < static_cast<int>(sArray[2]))
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_less(uArray[2],sArray[2]))

    if ((unsigned int)uArray[3] < (int)sArray[3])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_less(uArray[3],sArray[3]))

    if ((unsigned int)(uArray[4]) < (int)(sArray[4]))
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_less((uArray[4]),(sArray[4])))

    if (uArray[5] > sArray[5])
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_greater(uArray[5] , sArray[5]))

    #define VALUE sArray[6]
    if (uArray[6] > VALUE)
        return 0;
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: comparison between 'signed' and 'unsigned' integers [modernize-use-integer-sign-comparison]
// CHECK-FIXES: if (q20::cmp_greater(uArray[6] , VALUE))


    FuncParameters(uVar);
    TemplateFuncParameter(sVar);
    TemplateFuncParameters(uVar, sVar);

    return 0;
}
