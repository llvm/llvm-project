// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t -- -config="{CheckOptions: {modernize-use-init-statement.IgnoreConditionVariableStatements: 'true'}}"

void do_some(int i=0);

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

