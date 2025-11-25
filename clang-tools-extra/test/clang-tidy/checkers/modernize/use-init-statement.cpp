// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t

void do_some();

void normal() {
    int i1 = 0;
    // TODO: implement cases with and without side effects here;
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
void bad1() {
    int i1 = 0;
    if (i1 == 0) {
        do_some();
    }
    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            break;
    }
}

void bad2() {
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        ++i1;
    }
    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            ++i2;
            break;
    }
}
