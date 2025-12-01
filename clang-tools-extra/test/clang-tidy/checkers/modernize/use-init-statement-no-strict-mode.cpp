// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t -- -config="{CheckOptions: {modernize-use-init-statement.StrictMode: 'false'}}"

void do_some(int i=0);
#define DUMMY_TOKEN // crutch because CHECK-FIXES unable to match empty string

void good() {
    {
        {
            int i1 = 0;
            if (i1 == 0) {
                do_some();
            }
        }

        int i2 = 0;
        switch (i2) {
            case 0:
                do_some();
                break;
        }
    }
    do_some();
}

void bad1() {
    int i1=0, k1=0, j1=0; DUMMY_TOKEN
    if (i1 == 0 && k1 == 0 && j1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1=0, k1=0, j1=0; i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
    }
    do_some(); // Additional statement after if

    int i2=0, k2=0, j2=0; DUMMY_TOKEN
    switch (i2+k2+j2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2=0, k2=0, j2=0; i2+k2+j2) {
        case 0:
            do_some();
            break;
    }
    do_some(); // Additional statement after switch
}

void bad2() {
    {
        {
            int i1 = 0; DUMMY_TOKEN
            if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:13: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; i1 == 0) {
                do_some();
            }
            do_some(); // Not last in inner compound
        }

        int i2 = 0; DUMMY_TOKEN
        switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:9: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; i2) {
            case 0:
                do_some();
                break;
        }
        do_some(); // Not last in inner compound
    }
    // Outer compound ends here
}

