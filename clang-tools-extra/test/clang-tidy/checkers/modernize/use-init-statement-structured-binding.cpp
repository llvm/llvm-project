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
    class string_view {
    public:
        string_view() {}
        string_view(const string&) {}
        bool empty() const { return true; }
    };
    template<typename T, typename U>
    struct pair {
        T first;
        U second;
        pair() : first(), second() {}
    };
}
#define DUMMY_TOKEN // crutch because CHECK-FIXES unable to match empty string

struct Point {
    int x, y;
    Point() : x(0), y(0) {}
};

struct TupleLike {
    int a, b, c;
    TupleLike() : a(0), b(0), c(0) {}
};

template<typename T> const T* get_pointer(const T& ref) { return &ref; }

struct UserDefined {
    int a = 0;
    const UserDefined* get_pointer_to_this() const {
        return this;
    }
};

int get_temporary() { return 0; }

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

template<typename T>
void good_template() {
    auto [a1, b1] = T{};
    if (a1 == 0) {
        do_some();
    }
    ++b1; // Used after if, so don't suggest moving

    auto [a2, b2] = T{};
    switch (a2) {
        case 0:
            do_some();
            break;
    }
    ++b2; // Used after switch, so don't suggest moving
}

template<typename T>
struct TEMPLATE_STRUCT {
void good_template2() {
    auto [a1, b1] = T{};
    if (a1 == 0) {
        do_some();
    }
    ++b1; // Used after if, so don't suggest moving

    auto [a2, b2] = T{};
    switch (a2) {
        case 0:
            do_some();
            break;
    }
    ++b2; // Used after switch, so don't suggest moving
}
};

void good_already_has_init_stmt() {
    int arr[2] = {1, 2};
    auto [a, b] = arr;
    if (int i = 0; a == 0) {
        do_some();
    }
}

// FIXME: enable it
// void good_unused() {
//     int i =0;

//     // doesn't make sence to fix, this is a job for another checker
//     int arr2[2] = {1, 2};
//     auto [a2, b2] = arr2;
//     if (i == 0) {}

//     // same
//     int arr3[2] = {1, 2};
//     auto [a3, b3] = arr3;
//     switch (i) {
//         case 0:
//             break;
//     }
// }

// void good_unused_multiple() {
//     int i = 0;

//     // doesn't make sence to fix, this is a job for another checker
//     int arr3[2] = {1, 2};
//     auto [a3, b3] = arr3;
//     if (i == 0) {}

//     // same
//     int arr4[2] = {1, 2};
//     auto [a4, b4] = arr4;
//     switch (i) {
//         case 0:
//             break;
//     }
// }


using unique_lock_t = std::unique_lock<std::mutex>;

void good_unique_lock() {
    struct Locks { unique_lock_t l; };
    static std::mutex lock;
    static int counter = 0;
    auto [l] = Locks{unique_lock_t{lock}};
    if (l.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_lifetime_extension() {
    struct Locks { unique_lock_t l; };
    static std::mutex lock;
    static int counter = 0;
    const auto& [l] = Locks{unique_lock_t{lock}};
    if (l.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_multiple() {
    struct Locks { unique_lock_t first; unique_lock_t second; };
    static std::mutex lock1;
    static int counter1 = 0;

    static std::mutex lock2;
    static int counter2 = 0;

    auto locks = Locks{unique_lock_t{lock1}, unique_lock_t{lock2}};
    auto [l1, l2] = locks;
    if (l1.owns_lock()) {
        do_some();
    }
    ++counter1;
}

void good_unique_lock_multiple_different() {
    struct Locks { unique_lock_t l; int value; };
    static std::mutex lock;
    static int counter = 0;

    auto locks = Locks{unique_lock_t{lock}, 100};
    auto [l, value] = locks;
    if (l.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_const() {
    struct Locks { unique_lock_t l; };
    static std::mutex lock;
    static int counter = 0;
    const auto [l] = Locks{unique_lock_t{lock}};
    if (l.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_multiple_const() {
    struct Locks { unique_lock_t first; unique_lock_t second; };
    static std::mutex lock1;
    static int counter1 = 0;

    static std::mutex lock2;
    static int counter2 = 0;

    auto locks = Locks{unique_lock_t{lock1}, unique_lock_t{lock2}};
    const auto [l1, l2] = locks;
    if (l1.owns_lock()) {
        do_some();
    }
    ++counter1;
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

void good_stolen_reference1() {
    const int* pi = nullptr;
    int arr[2] = {0, 1};
    auto [i1, k1] = arr;
    if (i1 == 0) {
        do_some();
        pi = &i1;
    }
    do_some(*pi);

    int arr2[2] = {0, 1};
    auto [i2, k2] = arr2;
    switch (i2) {
        case 0:
            do_some();
            pi = &i2;
            break;
    }
    do_some(*pi);
}

void good_stolen_reference1_multiple() {
    const int* pi = nullptr;
    int arr[3] = {0, 0, 0};
    auto [i1, k1, j1] = arr;
    if (i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
        pi = &i1;
    }
    do_some(*pi);

    int arr2[3] = {0, 0, 0};
    auto [i2, k2, j2] = arr2;
    switch (i2 + k2 + j2) {
        case 0:
            do_some();
            pi = &i2;
            break;
    }
    do_some(*pi);
}

void good_stolen_reference2() {
    const int* pi = nullptr;
    int arr[2] = {0, 1};
    auto [i1, k1] = arr;
    if (i1 == 0) {
        do_some();
        pi = get_pointer(i1);
    }
    do_some(*pi);

    int arr2[2] = {0, 1};
    auto [i2, k2] = arr2;
    switch (i2) {
        case 0:
            do_some();
            pi = get_pointer(i2);
            break;
    }
    do_some(*pi);
}

void good_stolen_reference_as_this() {
    const UserDefined* pa = nullptr;
    struct Pair { UserDefined first; int second; };
    Pair p1{UserDefined{}, 0};
    auto [a, unused1] = p1;
    if (a.a == 0) {
        do_some();
        pa = a.get_pointer_to_this();
    }
    do_some(pa->a);

    Pair p2{UserDefined{}, 0};
    auto [b, unused2] = p2;
    switch (b.a) {
        case 0:
            do_some();
            pa = b.get_pointer_to_this();
            break;
    }
    do_some(pa->a);
}

void good_stolen_reference1_string() {
    const std::string* ps = nullptr;
    std::pair<std::string, int> p1;
    auto [s1, unused1] = p1;
    if (s1.empty()) {
        do_some();
        ps = &s1;
    }
    ps->empty();
}

void good_stolen_reference2_string() {
    std::string_view sv;
    std::pair<std::string, int> p1;
    auto [s1, unused1] = p1;
    if (s1.empty()) {
        do_some();
        sv = s1;
    }
    sv.empty();
}

void bad_stolen_reference2() {
    const int* pi = nullptr;
    int arr[2] = {0, 1};
    const auto& [i1, k1] = arr; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto& [i1, k1] = arr; i1 == 0) {
        do_some();
        pi = &i1;
    }
    do_some(*pi);

    int arr2[2] = {0, 1};
    const auto& [i2, k2] = arr2; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const auto& [i2, k2] = arr2; i2) {
        case 0:
            do_some();
            pi = &i2;
            break;
    }
    do_some(*pi);
}

void bad_prevents_redeclaration1() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr;
    if (int a1 = arr[0]) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }

    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2;
    switch (int b2 = arr[1])
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
    {
    case 0:
        do_some();
        break;
    }
}

void bad_prevents_redeclaration2() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr;
    if (a1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        int arr_inner[2] = {1, 2};
        auto [a1, b1] = arr_inner;
        do_some(a1);
    }

    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2;
    switch (a2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            int arr_inner[2] = {1, 2};
            auto [a2, b2] = arr_inner;
            do_some(a2);
            break;
        }
    }
}

void bad_prevents_redeclaration3() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr;
    if (a1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    } else {
        int arr_inner[2] = {1, 2};
        auto [a1, b1] = arr_inner;
        do_some(a1);
    }

    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2;
    switch (a2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            do_some();
            break;
        }
        case 1: {
            int arr_inner[2] = {1, 2};
            auto [a2, b2] = arr_inner;
            do_some(a2);
            break;
        }
    }
}

void bad_prevents_redeclaration4() {
    int arr[2] = {1, 2};
    auto [a1, b1] = arr;
    if (a1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        int arr_inner1[2] = {1, 2};
        auto [a1, b1] = arr_inner1;
        do_some(a1+1);
    } else {
        int arr_inner2[2] = {1, 2};
        auto [a1, b1] = arr_inner2;
        do_some(a1+2);
    }

    int arr2[2] = {1, 2};
    auto [a2, b2] = arr2;
    switch (a2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            int arr_inner1[2] = {1, 2};
            auto [a2, b2] = arr_inner1;
            do_some(a2+1);
            break;
        }
        case 1: {
            int arr_inner2[2] = {1, 2};
            auto [a2, b2] = arr_inner2;
            do_some(a2+2);
            break;
        }
    }
}


// FIXME: the same test but for `auto [str, val]`
void bad_safe_string_default() {
    std::pair<std::string, int> p;
    const auto& [str, val] = p; DUMMY_TOKEN
    if (str.empty()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto& [str, val] = p; str.empty()) {
        do_some();
    }
    do_some(); // Additional statement after if
}

// FIXME: the same test but for `auto [str, val]`
void bad_safe_string_default2() {
    struct P {std::string first; int second;};
    P p{{}, 0};
    const auto& [str, val] = p; DUMMY_TOKEN
    if (str.empty()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto& [str, val] = p; str.empty()) {
        do_some();
    }
    do_some(); // Additional statement after if
}

void bad_lifetime_extension_of_builtin_structured_binding() {
    struct IntWrapper { int value; };
    const auto& [i1] = IntWrapper{0}; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto& [i1] = IntWrapper{0}; i1 == 0) {
        do_some();
    }
    const auto& [i2] = IntWrapper{0}; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: structured binding declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const auto& [i2] = IntWrapper{0}; i2) {
        case 0:
            do_some();
            break;
    }
    do_some();
}

