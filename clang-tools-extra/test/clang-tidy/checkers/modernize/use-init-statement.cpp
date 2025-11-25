// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t

void do_some();
int get_with_possible_side_effects();

void normal() {
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

// TODO: implement structured binding case
// TODO: implement case for already if-init statement used
// TODO: implement case for const and constexpr variables
// TODO: implement case for multiple variables in one line
void bad1() {
    int i1 = 0;
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: if (int i1 = 0; i1 == 0) {
        do_some();
    }
    int i2 = 0;
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: switch (int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad2() {
    int i1 = 0;
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: if (int i1 = 0; i1 == 0) {
        do_some();
        ++i1;
    }
    int i2 = 0;
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: switch (int i2 = 0; i2) {
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
