// RUN: %check_clang_tidy %s readability-block-size %t \
// RUN: -config='{CheckOptions: { \
// RUN:  readability-block-size.IfLineCountThreshold: 5, \
// RUN:  readability-block-size.ForLineCountThreshold: 6, \
// RUN:  readability-block-size.WhileLineCountThreshold: 7 }}'

void should_warn(){
    if (true){ // 1
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: if block spans 6 lines of code, which exceeds the threshold of 5 lines [readability-block-size]
        int sum = 3
                + 4
                + 5;
    } //          6


    if (true){ // 1
        int sum = 2;
    } else { //   3     1
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: else block spans 6 lines of code, which exceeds the threshold of 5 lines [readability-block-size]
        int sum = 5  // 3
                + 6  // 4
                + 7; // 5
    } //          8     6


    if (true){ // 1
        int sum = 2;
    } else if (true){ // 1
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: if block spans 6 lines of code, which exceeds the threshold of 5 lines [readability-block-size]
        int sum = 5   // 3
                + 6   // 4
                + 7;  // 5
    } else { //   8      6
        int sum = 9;
    } //          10


    for (int i = 0; i < 10; ++i) {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: for loop spans 7 lines of code, which exceeds the threshold of 6 lines [readability-block-size]
        int sum = 3
                + 4
                + 5
                + 6;
    }

    while (true) {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: while loop spans 8 lines of code, which exceeds the threshold of 7 lines [readability-block-size]
        int sum = 3
                + 4
                + 5
                + 6
                + 7;
    }
}

void should_not_warn(){
    if (true){ // 1
        int sum = 2
                + 3
                + 4;
    } //          5

    bool a = true;
    bool b = false;
    if (a && b){
        int sum = 2
                + 3;
    } else if (a || b) {
        int sum = 5
                + 6
                + 7;
    } else {
        int sum = 9
                + 10;
    }


    for (int i = 0; i < 10; ++i) {
        int sum = 2
                + 3
                + 4
                + 5;
    }

    while (true) {
        int sum = 2
                + 3
                + 4
                + 5
                + 6;
    }
}
