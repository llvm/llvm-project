// RUN: %check_clang_tidy %s readability-if-block-size %t

void should_warn(){
    if (true){ // 1
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: if block spans 21 lines of code, which exceeds the threshold of 20 lines [readability-if-block-size]
        int sum = 3
                + 4
                + 5
                + 6
                + 7
                + 8
                + 9
                + 10
                + 11
                + 12
                + 13
                + 14
                + 15
                + 16
                + 17
                + 18
                + 19
                + 20;
    } //          21


    if (true){ // 1
        int sum = 2;
    } else { //   3      1
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: else block spans 21 lines of code, which exceeds the threshold of 20 lines [readability-if-block-size]
        int sum = 5   // 3
                + 6   // 4
                + 7   // 5
                + 8   // 6
                + 9   // 7
                + 10  // 8
                + 11  // 9
                + 12  // 10
                + 13  // 11
                + 14  // 12
                + 15  // 13
                + 16  // 14
                + 17  // 15
                + 18  // 16
                + 19  // 17
                + 20  // 18
                + 21  // 19
                + 22; // 20
    } //          23     21


    if (true){ // 1
        int sum = 2;
    } else if (true){ // 1
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: if block spans 21 lines of code, which exceeds the threshold of 20 lines [readability-if-block-size]
        int sum = 5   // 3
                + 6   // 4
                + 7   // 5
                + 8   // 6
                + 9   // 7
                + 10  // 8
                + 11  // 9
                + 12  // 10
                + 13  // 11
                + 14  // 12
                + 15  // 13
                + 16  // 14
                + 17  // 15
                + 18  // 16
                + 19  // 17
                + 20  // 18
                + 21  // 19
                + 22; // 20
    } else { //   23     21
        int sum = 24;
    } //          25
}

void should_not_warn(){
    if (true){ // 1
        int sum = 2
                + 3
                + 4
                + 5
                + 6
                + 7
                + 8
                + 9
                + 10
                + 11
                + 12
                + 13
                + 14
                + 15
                + 16
                + 17
                + 18
                + 19;
    } //          20

    bool a = true;
    bool b = false;
    if (a && b){
        int sum = 2
                + 3
                + 4
                + 5
                + 6;
    } else if (a || b) {
        int sum = 8
                + 9
                + 10
                + 11
                + 12
                + 13
                + 14
                + 15;
    } else {
        int sum = 17
                + 18
                + 19
                + 20
                + 21;
    }
}
