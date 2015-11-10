// RUN: %check_clang_tidy %s misc-macro-parentheses %t

#define BAD1              -1
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: macro replacement list should be enclosed in parentheses [misc-macro-parentheses]
#define BAD2              1+2
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: macro replacement list should be enclosed in parentheses [misc-macro-parentheses]
#define BAD3(A)           (A+1)
// CHECK-MESSAGES: :[[@LINE-1]]:28: warning: macro argument should be enclosed in parentheses [misc-macro-parentheses]
#define BAD4(x)           ((unsigned char)(x & 0xff))
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: macro argument should be enclosed in parentheses [misc-macro-parentheses]

#define GOOD1             1
#define GOOD2             (1+2)
#define GOOD3(A)          #A
#define GOOD4(A,B)        A ## B
#define GOOD5(T)          ((T*)0)
#define GOOD6(B)          "A" B "C"
#define GOOD7(b)          A b
#define GOOD8(a)          a B
#define GOOD9(type)       (type(123))
#define GOOD10(car, ...)  car
#define GOOD11            a[b+c]
#define GOOD12(x)         a[x]
#define GOOD13(x)         a.x
#define GOOD14(x)         a->x
#define GOOD15(x)         ({ int a = x; a+4; })
#define GOOD16(x)         a_ ## x, b_ ## x = c_ ## x - 1,
#define GOOD17            case 123: x=4+5; break;
#define GOOD18(x)         ;x;
#define GOOD19            ;-2;
#define GOOD20            void*
#define GOOD21(a)         case Fred::a:
#define GOOD22(a)         if (verbose) return a;
#define GOOD23(type)      (type::Field)
#define GOOD24(t)         std::set<t> s
#define GOOD25(t)         std::set<t,t,t> s
#define GOOD26(x)         (a->*x)
#define GOOD27(x)         (a.*x)

// These are allowed for now..
#define MAYBE1            *12.34
#define MAYBE2            <<3
