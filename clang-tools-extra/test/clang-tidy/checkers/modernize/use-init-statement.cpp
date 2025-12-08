// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-init-statement %t -- -- -I %S/Inputs/use-init-statement

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
        string& operator=(const char*) { return *this; }
        bool empty() const { return true; }
    };
    class string_view {
    public:
        string_view() {}
        string_view(const string&) {}
        string_view& operator=(const string&) { return *this; }
        bool empty() const { return true; }
    };
}
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

template<typename T>
void good_template() {
    T i1{};
    if (i1 == 0) {
        do_some();
    }
    ++i1;
    T i2{};
    switch (i2) {
        case 0:
            do_some();
            break;
    }
    ++i2;
}

template<typename T>
struct TEMPLATE_STRUCT {
void good_template2() {
    T i1{};
    if (i1 == 0) {
        do_some();
    }
    ++i1;
    T i2{};
    switch (i2) {
        case 0:
            do_some();
            break;
    }
    ++i2;
}
};

void good_already_has_init_stmt() {
    int i1 = 0;
    if (int i=0; i1 == 0) {
// FIXME: convert to bad case - should be changed to 'if (int i1=0,i=0; i1 == 0) {'
        do_some(i);
    }
    int i2 = 0;
    switch (int i=0; i2) {
// FIXME: convert to bad case - should be changed to 'switch (int i2=0,i=0; i2) {'
        case 0:
            do_some(i);
            break;
    }
}

void good_multiple() {
    int i1=0, k1=0, j1=0;
    if (i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
    }
    ++k1;
    int i2=0, k2=0, j2=0;
    switch (i2+k2+j2) {
        case 0:
            do_some();
            break;
    }
    ++j2;
}

void good_prev_decl_stmt_not_a_variable() {
    struct S1 { bool operator==(int); };
    if (S1{} == 0) {
        do_some();
    }
}

void good_unique_lock() {
    static std::mutex counter_mutex;
    static int counter;

    std::unique_lock<std::mutex> lock(counter_mutex);
    if (lock.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_lifetime_extension() {
    static std::mutex counter_mutex;
    static int counter;

    const auto& lock = std::unique_lock<std::mutex>{counter_mutex};
    if (lock.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_nested() {
    struct Lock { std::unique_lock<std::mutex> l; };
    static std::mutex counter_mutex;
    static int counter;

    Lock lock{{counter_mutex}};
    if (lock.l.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_multiple() {
    static std::mutex counter_mutex;
    static int counter;

    static std::mutex counter2_mutex;
    static int counter_2;

    std::unique_lock<std::mutex> lock(counter_mutex);
    std::unique_lock<std::mutex> lock2(counter2_mutex), *p_loc = &lock;
    if (lock2.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_array_of_unique_lock() {
    static std::mutex counter_mutex;
    static int counter;

    static std::mutex counter2_mutex;
    static int counter2;

    std::unique_lock<std::mutex> lock[2] = {
        {counter_mutex},
        {counter2_mutex}
    };
    if (lock[0].owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unique_lock_using() {
    using LockType = std::unique_lock<std::mutex>;
    static std::mutex counter_mutex;
    static int counter;

    LockType lock(counter_mutex);
    if (lock.owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_array_of_unique_lock_using() {
    using LockType = std::unique_lock<std::mutex>;
    static std::mutex counter_mutex;
    static int counter;

    static std::mutex counter2_mutex;
    static int counter2;

    LockType lock[2] = {
        {counter_mutex},
        {counter2_mutex}
    };
    if (lock[0].owns_lock()) {
        do_some();
    }
    ++counter;
}

void good_unused() {
    int var = 0;

    // doesn't make sence to fix, this is a job for another checker
    int var2 = 0;
    if (var) {}

    // same
    int var3 = 0;
    switch (var) {
        case 0:
            break;
    }
}

void good_unused_multiple() {
    int var = 0;

    // doesn't make sence to fix, this is a job for another checker
    int var2 = 0, var3 = 0;
    if (var) {}

    // same
    int var4 = 0, var5 = 0;
    switch (var) {
        case 0:
            break;
    }
}

// Real-life case, got from clang/lib/Basic/Attributes.cpp
void good_include() {
    {
        int i1 = 0;
#include "separate_if.hpp"
    }
    {
        int i2 = 0;
#include "separate_switch.hpp"
    }
}

void good_stolen_reference1() {
    const int* pi = nullptr;
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        pi = &i1;
    }
    do_some(*pi);

    int i2 = 0;
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
    int i1 = 0, k1 = 0, j1 = 0;
    if (i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
        pi = &i1;
    }
    do_some(*pi);

    int i2 = 0, k2 = 0, j2 = 0;
    switch (i2 + k2 + j2) {
        case 0:
            do_some();
            pi = &i2;
            break;
    }
    do_some(*pi);
}

void good_stolen_reference1_const_int_ptr() {
    int value = 0;
    const int** ppi = nullptr;
    const int* pi1 = &value;
    if (pi1 != nullptr) {
        do_some();
        ppi = &pi1;
    }
    do_some(**ppi);

    const int* pi2 = &value;
    switch (pi2 != nullptr ? 0 : 1) {
        case 0:
            do_some();
            ppi = &pi2;
            break;
    }
    do_some(**ppi);
}

template<typename T> const T* get_pointer(const T& ref) { return &ref; }

void good_stolen_reference2() {
    const int* pi = nullptr;
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        pi = get_pointer(i1);
    }
    do_some(*pi);

    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            pi = get_pointer(i2);
            break;
    }
    do_some(*pi);
}

struct ValueConverter {
    const int* get_pointer(const int& ref) const {
        return &ref;
    }
    static const int* get_pointer_stat(const int& ref) {
        return &ref;
    }
};

void good_stolen_reference2_member() {
    const int* pi = nullptr;
    ValueConverter cnv;
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        pi = cnv.get_pointer(i1);
    }
    do_some(*pi);

    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            pi = cnv.get_pointer(i2);
            break;
    }
    do_some(*pi);
}

void good_stolen_reference2_static_member() {
    const int* pi = nullptr;
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        pi = ValueConverter::get_pointer_stat(i1);
    }
    do_some(*pi);

    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            pi = ValueConverter::get_pointer_stat(i2);
            break;
    }
    do_some(*pi);
}

struct PointerGetter {
    const int* operator()(const int& ref) const {
        return &ref;
    }
};

void good_stolen_reference2_operator() {
    const int* pi = nullptr;
    PointerGetter getter;
    int i1 = 0;
    if (i1 == 0) {
        do_some();
        pi = getter(i1);
    }
    do_some(*pi);

    int i2 = 0;
    switch (i2) {
        case 0:
            do_some();
            pi = getter(i2);
            break;
    }
    do_some(*pi);
}

struct UserDefined {
    int a = 0;
    const UserDefined* get_pointer_to_this() const {
        return this;
    }
};

void good_stolen_reference_as_this() {
    const UserDefined* pa = nullptr;
    UserDefined a;
    if (a.a == 0) {
        do_some();
        pa = a.get_pointer_to_this();
    }
    do_some(pa->a);

    UserDefined b;
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
    std::string s1;
    if (s1.empty()) {
        do_some();
        ps = &s1;
    }
    ps->empty();
}

void good_stolen_reference2_string() {
    std::string_view sv;
    std::string s1;
    if (s1.empty()) {
        do_some();
        sv = s1;
    }
    sv.empty();
}

// Real-life case, got from LSPClient.cpp and CompileCommands.cpp
void good_stringref_storage_lifetime() {
    std::string_view path;
    std::string Storage;
    // Storage is assigned inside if, and path is reassigned to point to Storage
    // This matches the pattern: if (!is_absolute(Path)) Path = Storage = testPath(Path);
    // The key is that Storage must outlive the if because path points to it after the if
    if (Storage.empty()) {  // condition can use Storage or path
        path = Storage = "absolute_path";
    }
    // path is used after if, so Storage must outlive the if statement
    path.empty();
    
    // Same pattern with switch
    std::string_view path2;
    std::string Storage2;
    switch (Storage2.empty() ? 0 : 1) {
        case 0:
            path2 = Storage2 = "absolute_path2";
            break;
    }
    path2.empty();
}

int get_temporary() { return 0; }

void good_stolen_temporary_materialized_reference() {
    const int* pi = nullptr;
    const int& ref1 = get_temporary();
    if (ref1 == 0) {
        do_some();
        pi = &ref1;
    }
    do_some(*pi);

    const int& ref2 = get_temporary();
    switch (ref2) {
        case 0:
            do_some();
            pi = &ref2;
            break;
    }
    do_some(*pi);
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

void bad_unused_in_condition() {
    int i = 0;

    int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; i1 == 0) {
        // 'i1' unable to be placed here by another check
        do_some(i1);
    } else {
        // 'i1' unable to be placed here by another check
        do_some(i1+1);
    }

    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; i2) {
        case 0: {
            // 'i2' unable to be placed here by another check
            do_some(i2);
            break;
        }
        case 1: {
            // 'i2' unable to be placed here by another check
            do_some(i2+1);
            break;
        }
    }
}

void bad_const() {
    const int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const int i1 = 0; i1 == 0) {
        do_some();
    }
    const int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_volatile() {
    volatile int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (volatile int i1 = 0; i1 == 0) {
        do_some();
    }
    volatile int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (volatile int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_const_volatile() {
    const volatile int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const volatile int i1 = 0; i1 == 0) {
        do_some();
    }
    const volatile int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const volatile int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_constexpr() {
    constexpr int i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (constexpr int i1 = 0; i1 == 0) {
        do_some();
    }
    constexpr int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (constexpr int i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_unitialized() {
    int i1; DUMMY_TOKEN
    if ((i1=0) == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1; (i1=0) == 0) {
        do_some();
    }
    int i2; DUMMY_TOKEN
    switch (i2=0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2; i2=0) {
        case 0:
            do_some();
            break;
    }
}

void bad_multiple() {
    int i1=0, k1=0, j1=0; DUMMY_TOKEN
    if (i1 == 0 && k1 == 0 && j1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1=0, k1=0, j1=0; i1 == 0 && k1 == 0 && j1 == 0) {
        do_some();
    }
    int i2=0, k2=0, j2=0; DUMMY_TOKEN
    switch (i2+k2+j2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2=0, k2=0, j2=0; i2+k2+j2) {
        case 0:
            do_some();
            break;
    }
}

void bad_multiple_not_all_used() {
    int i1=0, k1=0, j1=0; DUMMY_TOKEN
    if (i1 == 0 && k1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1=0, k1=0, j1=0; i1 == 0 && k1 == 0) {
        do_some();
    }
    int i2=0, k2=0, j2=0; DUMMY_TOKEN
    switch (i2+j2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: multiple variable declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2=0, k2=0, j2=0; i2+j2) {
        case 0:
            do_some();
            break;
    }
}

void bad_lifetime_extension_of_builtin() {
    const int& i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const int& i1 = 0; i1 == 0) {
        do_some();
    }
    const int& i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const int& i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
    do_some();
}

void bad_pointer_to_unique_lock() {
    static std::mutex counter_mutex;
    static int counter;

    std::unique_lock<std::mutex> lock(counter_mutex);

    ++counter;

    const auto* p_lock = &lock; DUMMY_TOKEN
    if (p_lock->owns_lock()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'p_lock' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto* p_lock = &lock; p_lock->owns_lock()) {
        do_some();
    }
    ++counter;
}

void bad_reference_to_unique_lock() {
    static std::mutex counter_mutex;
    static int counter;

    std::unique_lock<std::mutex> lock(counter_mutex);

    ++counter;
    
    const auto& r_lock = lock; DUMMY_TOKEN
    if (r_lock.owns_lock()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'r_lock' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const auto& r_lock = lock; r_lock.owns_lock()) {
        do_some();
    }
    ++counter;
}

void bad_pointer_to_unique_lock_using() {
    using LockType = std::unique_lock<std::mutex>;
    static std::mutex counter_mutex;
    static int counter;

    LockType lock(counter_mutex);

    ++counter;

    using LockPtr = const LockType*;
    LockPtr p_lock = &lock; DUMMY_TOKEN
    if (p_lock->owns_lock()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'p_lock' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (LockPtr p_lock = &lock; p_lock->owns_lock()) {
        do_some();
    }
    ++counter;
}

void bad_reference_to_unique_lock_using() {
    using LockType = std::unique_lock<std::mutex>;
    static std::mutex counter_mutex;
    static int counter;

    LockType lock(counter_mutex);

    ++counter;
    
    const LockType& r_lock = lock; DUMMY_TOKEN
    if (r_lock.owns_lock()) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'r_lock' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const LockType& r_lock = lock; r_lock.owns_lock()) {
        do_some();
    }
    ++counter;
}

void bad_condition_with_declaration() {
    int i1 = 0; DUMMY_TOKEN
    if (int j = i1+1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; int j = i1+1) {
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch (int j = i2+1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; int j = i2+1) {
        case 0:
            do_some();
            break;
    }
}

void bad_prevents_redeclaration1() {
    int i1 = 0;
    if (int i1 = 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }

    int i2 = 0;
    switch (int i2 = 0)
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
    {
    case 0:
        do_some();
        break;
    }
}

void bad_prevents_redeclaration2() {
    int i1 = 0;
    if (i1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        int i1 = 0;
        do_some(i1);
    }

    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            int i2 = 0;
            do_some(i2);
            break;
        }
    }
}

void bad_prevents_redeclaration3() {
    int i1 = 0;
    if (i1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    } else {
        int i1 = 0;
        do_some(i1);
    }

    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            do_some();
            break;
        }
        case 1: {
            int i2 = 0;
            do_some(i2);
            break;
        }
    }
}

void bad_prevents_redeclaration4() {
    int i1 = 0;
    if (i1) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        int i1 = 0;
        do_some(i1+1);
    } else {
        int i1 = 0;
        do_some(i1+2);
    }

    int i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0: {
            int i2 = 0;
            do_some(i2+1);
            break;
        }
        case 1: {
            int i2 = 0;
            do_some(i2+2);
            break;
        }
    }
}

void bad_stolen_reference1() {
    const int* pi = nullptr;
    int val = 0;
    const int& ref1 = val; DUMMY_TOKEN
    if (ref1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'ref1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (const int& ref1 = val; ref1 == 0) {
        do_some();
        pi = &ref1;
    }
    do_some(*pi);

    const int& ref2 = val; DUMMY_TOKEN
    switch (ref2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'ref2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (const int& ref2 = val; ref2) {
        case 0:
            do_some();
            pi = &ref2;
            break;
    }
    do_some(*pi);
}

#define OPEN_PAREN_I1 (i1
#define OPEN_PAREN_I2 (i2
#define OPEN_PAREN_F() (
#define CLOSE_PAREN )
#define SEMICOLON_IF ; if
#define SEMICOLON_SWITCH ; switch
#define SEMICOLON_INT ; int
#define SEMICOLON ;
#define ZERO 0
#define MY_INT int
#define I1 i1
#define I2 i2
#define MY_IF(cond, stmt) if (cond) {stmt}
#define MY_ONE_STMT_SWITCH(tag, stmt) switch (tag) { case 0 : { stmt; break; } };
#define MY_VAR_DECL(type, name, val) type name = val

void bad_macro1() {
    int i1 = 0;
    if OPEN_PAREN_I1 == 0 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int i2 = 0;
    switch OPEN_PAREN_I2 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro2() {
    int i1 = 0 SEMICOLON_IF
     (i1 == 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    }
    int i2 = 0 SEMICOLON_SWITCH
     (i2) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        case 0:
            do_some();
            break;
    }
}

void bad_macro3() {
    int iprev1 = 0 SEMICOLON_INT
     i1 = 0;
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-3]]:20: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
        do_some();
    }
    int iprev2 = 0 SEMICOLON_INT
     i2 = 0;
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-3]]:20: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes
        case 0:
            do_some();
            break;
    }
}

void bad_macro4() {
    int i1 = 0 SEMICOLON DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0 SEMICOLON i1 == 0) {
        do_some();
    }
    int i2 = 0 SEMICOLON DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0 SEMICOLON i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro5() {
    int i1 = ZERO; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = ZERO; i1 == 0) {
        do_some();
    }
    int i2 = ZERO; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = ZERO; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro6() {
    MY_INT i1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (MY_INT i1 = 0; i1 == 0) {
        do_some();
    }
    MY_INT i2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (MY_INT i2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro7() {
    int I1 = 0; DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int I1 = 0; i1 == 0) {
        do_some();
    }
    int I2 = 0; DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int I2 = 0; i2) {
        case 0:
            do_some();
            break;
    }
}

void bad_macro9() {
    int i1 = 0; DUMMY_TOKEN
    if OPEN_PAREN_F()i1 == 0 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if OPEN_PAREN_F()int i1 = 0; i1 == 0 CLOSE_PAREN {
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch OPEN_PAREN_F()i2 CLOSE_PAREN {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch OPEN_PAREN_F()int i2 = 0; i2 CLOSE_PAREN {
        case 0:
            do_some();
            break;
    }
}

void bad_macro10() {
    int i1 = 0; DUMMY_TOKEN
    if (I1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (int i1 = 0; I1 == 0) {
        do_some();
    }
    int i2 = 0; DUMMY_TOKEN
    switch (I2) {
// CHECK-MESSAGES: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (int i2 = 0; I2) {
        case 0:
            do_some();
            break;
    }
}

// void bad_macro11() {
//     int i1 = 0;
//     MY_IF (i1 == 0,
// // FIXME: fixit should be suggested
//         do_some();
//     )
//     int i2 = 0;
//     MY_ONE_STMT_SWITCH (i2,
// // FIXME: fixit should be suggested
//         do_some();
//     )
// }

void bad_macro12() {
    MY_VAR_DECL(int, i1, 0); DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES: [[@LINE-2]]:17: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: if (MY_VAR_DECL(int, i1, 0); i1 == 0) {
        do_some();
    }
    MY_VAR_DECL(int, i2, 0); DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES: [[@LINE-2]]:17: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
// CHECK-FIXES: DUMMY_TOKEN
// CHECK-FIXES-NEXT: switch (MY_VAR_DECL(int, i2, 0); i2) {
        case 0:
            do_some();
            break;
    }
}

#define MY_VAR_DECL_WITH_EXTRA(type, name, val) do_some(); type name = val;

void bad_macro12_extra() {
    MY_VAR_DECL_WITH_EXTRA(int, i1, 0); DUMMY_TOKEN
    if (i1 == 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    }
    MY_VAR_DECL_WITH_EXTRA(int, i2, 0); DUMMY_TOKEN
    switch (i2) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        case 0:
            do_some();
            break;
    }
}

void bad_macro13() {
    MY_VAR_DECL(int, i1, 0);
    MY_IF (i1 == 0,
// CHECK-MESSAGES-NOT: [[@LINE-2]]:17: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    )
    MY_VAR_DECL(int, i2, 0);
    MY_ONE_STMT_SWITCH (i2,
// CHECK-MESSAGES-NOT: [[@LINE-2]]:17: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        do_some();
    )
}

#define MY_VAR_DECL_IF(type, name, val, cond, stmt) MY_VAR_DECL(type, name, val); MY_IF(cond, stmt)
#define MY_VAR_DECL_SWITCH(type, name, val, tag, stmt) MY_VAR_DECL(type, name, val); MY_ONE_STMT_SWITCH(tag, stmt)

void bad_macro14() {
    MY_VAR_DECL_IF(int, i1, 0, i1 == 0,
// CHECK-MESSAGES-NOT: [[@LINE-1]]:20: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    )
    MY_VAR_DECL_SWITCH(int, i2, 0, i2,
// CHECK-MESSAGES-NOT: [[@LINE-1]]:24: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        do_some();
    )
}

static constexpr int affinity_none = 0;

#define KMP_WARNING(...) do_some(__VA_ARGS__)
#define KMP_AFF_WARNING(s, ...)                                                \
  if (s.flags.verbose || (s.flags.warnings && (s.type != affinity_none))) {    \
    KMP_WARNING(__VA_ARGS__);                                                  \
  }

// Real-life case, got from openmp/runtime/src/kmp_affinity.cpp
void bad_macro_kmp_warning() {
    struct kmp_affinity_flags {
        unsigned verbose : 1;
        unsigned warnings : 1;
    };
    struct kmp_affinity_t { kmp_affinity_flags flags{}; int type{}; };
    kmp_affinity_t __kmp_affinity{};
    int num = 0;
    bool plural = (num > 1);
    KMP_AFF_WARNING(__kmp_affinity, plural);
    // CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'plural' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
}

#define SOME_MACRO 0

void bad_macro_ifdef_statement_1() {
    int i1 = 0;
#if SOME_MACRO
    if (i1 == 0) {
        do_some();
    }
#endif
    if (i1 == 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    }

    int i2 = 0;
#if SOME_MACRO
    switch (i2) {
        case 0:
            do_some();
            break;
    }
#endif
    switch (i2) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        case 0:
            do_some();
            break;
    }
}

void bad_macro_ifdef_statement_2() {
    int i1 = 0;
    if (i1 == 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement [modernize-use-init-statement]
        do_some();
    }
#if SOME_MACRO
    ++i1;
#endif
    int i2 = 0;
    switch (i2) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init statement [modernize-use-init-statement]
        case 0:
            do_some();
            break;
    }
#if SOME_MACRO
    ++i2;
#endif
}

#define ASSERT(expr)

void bad_macro_assert() {
    int i1 = 0;
    ASSERT(i1 != 0);
    if (i1 == 0) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i1' declaration before if statement could be moved into if init statement
        do_some();
    }
    int i2 = 0;
    ASSERT(i2 != 0);
    switch (i2) {
// CHECK-MESSAGES-NOT: [[@LINE-2]]:5: warning: variable 'i2' declaration before switch statement could be moved into switch init
        case 0:
            do_some();
            break;
    }
}


