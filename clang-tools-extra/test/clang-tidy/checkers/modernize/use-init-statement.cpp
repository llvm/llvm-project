// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t

void do_some(int i=0);
int get_with_possible_side_effects();
enum class INITIALIZE_STATUS { OK, FAIL, PENDING };
INITIALIZE_STATUS initialize(int& val);
#define DUMMY_TOKEN // crutch because CHECK-FIXES unable to match empty string

void good() {
    int i1 = 0;
    if (i1 == 0) {
        do_some();
    }
    ++i1;
    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            break;
    }
    ++i2;
}

void good_already_has_init_stmt() {
    int i1 = 0;
    if (int i=0; i1 == 0) {
// FIXME: convert to bad case - should be changed to 'if (int i1=0,i=0; i1 == 0) {'
        do_some(i);
    }
    int i2 = 0;
    switch (int i=0; i2) {
// FIXME: convert to bad case - should be changed to 'switch (int i2=0,i=0; i2) {'
        case 0:
            do_some(i);
            break;
    }
}

void good_unused_in_condition() {
    int i = 0;

    int i1 = 0;
    if (i == 0) {
        // 'i1' will be placed here by another check
        do_some(i1);
    }

    int i2 = 0;
    switch (i) {
        case 0: {
            // 'i2' will be placed here by another check
            do_some(i2);
            break;
        }
    }
}

void good_multiple() {
    int i1=0, k1=0, j1=0;
    if (i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
    }
    ++k1;
    int i2=0, k2=0, j2=0;
    switch (i2+k2+j2) {
        case 0:
            do_some();
            break;
    }
    ++j2;
}

// TODO: implement structured binding case

void bad1() {
    int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; i1 == 0) {
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad2() {
    int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; i1 == 0) {
        do_some();
        ++i1;
    }
    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; i2) {
        case 0:
            do_some();
            ++i2;
            break;
    }
}

// FIXME: implement this case
// void bad3() {
//     int i1 = 0;
//     int ii1 = 0;
//     if (i1 == 0) {
//         do_some();
//     }
//     int i2 = 0;
//     int ii2 = 0;
//     switch (i2) {
//         case 0:
//             do_some();
//             break;
//     }
// }

void bad_const() {
    const int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const int i1 = 0; i1 == 0) {
        do_some();
    }
    const int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_constexpr() {
    constexpr int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (constexpr int i1 = 0; i1 == 0) {
        do_some();
    }
    constexpr int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (constexpr int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_unitialized() {
    int i1; DUMMY_TOKEN
    if (initialize(i1) == INITIALIZE_STATUS::OK) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1; initialize(i1) == INITIALIZE_STATUS::OK) {
        do_some();
    }
    int i2; DUMMY_TOKEN
    switch (initialize(i2)) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2; initialize(i2)) {
        case INITIALIZE_STATUS::OK:
            do_some();
            break;
    }
}

void bad_multiple() {
    int i1=0, k1=0, j1=0; DUMMY_TOKEN
    if (i1 == 0 && k1 == 0 && j1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1=0, k1=0, j1=0; i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
    }
    int i2=0, k2=0, j2=0; DUMMY_TOKEN
    switch (i2+k2+j2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2=0, k2=0, j2=0; i2+k2+j2) {
        case 0:
            do_some();
            break;
    }
}

#define OPEN_PAREN_I1 (i1
#define OPEN_PAREN_I2 (i2
#define OPEN_PAREN_F() (
#define CLOSE_PAREN )
#define SEMICOLON_IF ; if
#define SEMICOLON_SWITCH ; switch
#define SEMICOLON_INT ; int
#define SEMICOLON ;
#define ZERO 0
#define MY_INT int
#define I1 i1
#define I2 i2


void bad_macro1() {
    int i1 = 0;
    if OPEN_PAREN_I1 == 0 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int i2 = 0;
    switch OPEN_PAREN_I2 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro2() {
    int i1 = 0 SEMICOLON_IF
     (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int i2 = 0 SEMICOLON_SWITCH
     (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro3() {
    int iprev1 = 0 SEMICOLON_INT
     i1 = 0;
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-3]]:20: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int iprev2 = 0 SEMICOLON_INT
     i2 = 0;
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-3]]:20: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro4() {
    int i1 = 0 SEMICOLON
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int i2 = 0 SEMICOLON
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro5() {
    int i1 = ZERO; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = ZERO; i1 == 0) {
        do_some();
    }
    int i2 = ZERO; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = ZERO; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro6() {
    MY_INT i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// FIXME: should be changed to 'if (MY_INT i1 = 0; i1 == 0) {'
        do_some();
    }
    MY_INT i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// FIXME: should be changed to 'switch (MY_INT i2 = 0; i2) {'
        case 0:
            do_some();
            break;
    }
}

void bad_macro7() {
    int I1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int I1 = 0; i1 == 0) {
        do_some();
    }
    int I2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int I2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro9() {
    int i1 = 0; DUMMY_TOKEN
    if OPEN_PAREN_F()i1 == 0 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if OPEN_PAREN_F()int i1 = 0; i1 == 0 CLOSE_PAREN {
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch OPEN_PAREN_F()i2 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch OPEN_PAREN_F()int i2 = 0; i2 CLOSE_PAREN {
        case 0:
            do_some();
            break;
    }
}

void bad_macro10() {
    int i1 = 0; DUMMY_TOKEN
    if (I1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// FIXME: should be changed to 'if (int i1 = 0; I1 == 0) {'
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch (I2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// FIXME: should be changed to 'switch (int i2 = 0; I2) {'
        case 0:
            do_some();
            break;
    }
}
