// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t -- -config="{CheckOptions: {modernize-use-init-statement.IgnoreConditionVariableStatements: 'true'}}"

void do_some(int i=0);
#define DUMMY_TOKEN // crutch because CHECK-FIXES unable to match empty string

int getY();
int getB();

void good_ignored_when_condition_variable_exists() {
    int x = 0;
    if (auto y = getY()) {
        do_some(x);
    }

    int a = 0;
    switch (auto b = getB()) {
        case 0:
            do_some(a);
            break;
    }
}

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
