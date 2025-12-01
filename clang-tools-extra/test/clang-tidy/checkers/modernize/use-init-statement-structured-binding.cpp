// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t

void do_some(int i=0);
int get_with_possible_side_effects();
namespace std {
    struct mutex {};
    template<typename Mutex>
    struct unique_lock {
        Mutex* m;
        unique_lock(Mutex& mutex) : m(&mutex) {}
        ~unique_lock();
        bool owns_lock() const noexcept { return true; }
    };
    class string {
    public:
        string() {}
        ~string() {}
        bool empty() const { return true; }
    };
    template<typename T, typename U>
    struct pair {
        T first;
        U second;
        pair() : first(), second() {}
        //~pair() {}
    };
}
#define DUMMY_TOKEN // crutch because CHECK-FIXES unable to match empty string

struct Point {
    int x, y;
    Point() : x(0), y(0) {}
    // ~Point() {}
};

struct TupleLike {
    int a, b, c;
    TupleLike() : a(0), b(0), c(0) {}
    // ~TupleLike() {}
};

void good() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr;
    if (a1 == 0) {
        do_some();
    }
    ++b1; // Used after if, so don't suggest moving

    auto [a2, b2] = arr;
    switch (a2) {
        case 0:
            do_some();
            break;
    }
    ++b2; // Used after switch, so don't suggest moving
}

void good_already_has_init_stmt() {
    int arr[2] = {1, 2};
    auto [a, b] = arr;
    if (int i = 0; a == 0) {
        do_some();
    }
}

void bad1() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr; DUMMY_TOKEN
    if (a1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto [a1, b1] = arr; a1 == 0) {
        do_some();
    }
    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2; DUMMY_TOKEN
    switch (a2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (auto [a2, b2] = arr2; a2) {
        case 0:
            do_some();
            break;
    }
}

void bad2() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr; DUMMY_TOKEN
    if (a1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto [a1, b1] = arr; a1 == 0) {
        do_some();
        ++a1;
    }
    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2; DUMMY_TOKEN
    switch (a2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (auto [a2, b2] = arr2; a2) {
        case 0:
            do_some();
            ++a2;
            break;
    }
}

void bad_multiple_bindings_in_condition() {
    int arr[2] = {1, 2};
    auto [a, b] = arr; DUMMY_TOKEN
    if (a == 0 && b == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto [a, b] = arr; a == 0 && b == 0) {
        do_some();
    }
    int arr2[3] = {1, 2, 3};
    auto [x, y, z] = arr2; DUMMY_TOKEN
    switch (x + y) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (auto [x, y, z] = arr2; x + y) {
        case 0:
            do_some();
            break;
    }
}

void bad_struct_member() {
    Point p;
    auto [x, y] = p; DUMMY_TOKEN
    if (x == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto [x, y] = p; x == 0) {
        do_some();
    }
    Point p2;
    auto [x2, y2] = p2; DUMMY_TOKEN
    switch (x2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (auto [x2, y2] = p2; x2) {
        case 0:
            do_some();
            break;
    }
}

void bad_with_reference() {
    int arr[2] = {1, 2};
    auto &[a, b] = arr; DUMMY_TOKEN
    if (a == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto &[a, b] = arr; a == 0) {
        do_some();
    }
}

void bad_with_const() {
    int arr[2] = {1, 2};
    const auto [a, b] = arr; DUMMY_TOKEN
    if (a == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto [a, b] = arr; a == 0) {
        do_some();
    }
}

void bad_unused_in_condition() {
    int arr[2] = {1, 2};
    auto [a, b] = arr; DUMMY_TOKEN
    if (arr[0] == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (auto [a, b] = arr; arr[0] == 0) {
        // 'a' and 'b' unable to be placed here by another check
        do_some(a);
    } else {
        // 'a' and 'b' unable to be placed here by another check
        do_some(b);
    }
    int arr2[2] = {1, 2};
    auto [x, y] = arr2; DUMMY_TOKEN
    switch (arr2[0]) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (auto [x, y] = arr2; arr2[0]) {
        case 0: {
            // 'x' and 'y' unable to be placed here by another check
            do_some(x);
            break;
        }
        case 1: {
            // 'x' and 'y' unable to be placed here by another check
            do_some(y);
            break;
        }
    }
}

// FIXME: enable it when std::pair will be one of default type in DefaultSafeDestructorTypes
// void bad_safe_string_default() {
//     std::pair<std::string, int> p;
//     auto [str, val] = p; DUMMY_TOKEN
//     if (str.empty()) {
// FIXME: fixit must be here
//         do_some();
//     }
//     do_some(); // Additional statement after if
// }


