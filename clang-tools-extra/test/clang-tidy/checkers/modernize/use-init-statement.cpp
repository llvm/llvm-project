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
// FIXME: should be changed to 'if (int i1=0,i=0; i1 == 0) {'
        do_some(i);
    }
    int i2 = 0;
    switch (int i=0; i2) {
// FIXME: should be changed to 'switch (int i2=0,i=0; i2) {'
        case 0:
            do_some(i);
            break;
    }
}


// TODO: implement structured binding case
// TODO: implement case for const and constexpr variables
// TODO: implement case for multiple variables in one line
// TODO: test for case when unused in condition but used in body (this is for abother check)
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
