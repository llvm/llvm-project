// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-eval-order %t

struct Food {
    int cnt{};
    int i() { return ++cnt; }
    bool b() { return ++cnt > 1; }
    int ci() const { return cnt; }
    int cb() const { return cnt > 1; }
    static int I() {return {};}
};

int I() {
    static int cnt{};
    return ++cnt;
}

struct Cat {
    Cat(int, bool) {}
};

void Dog(int, bool) {}

int copy(int i) { return i; }

void function() {
    Food f;
    const Food cf;
    const Food& crf{cf};
    Food& mutf{f};
    // Assume const member functions are fine
    Cat a(cf.ci(), crf.cb());
    Cat(f.ci(), f.cb());
    Dog(f.cb(), f.ci());
    // Assume static and global functions are fine
    Dog(f.I(), f.b());
    Dog(I(), I());
    // C++11 list-init is fine
    Cat{f.i(), f.b()};
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: Order of evaluation of constructor arguments is unspecified.
    Cat(f.i(), f.b());
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: Order of evaluation of function arguments is unspecified.
    Dog(copy(mutf.i()), f.b());
    // ... even if one is const
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: Order of evaluation of function arguments is unspecified.
    Dog(f.i(), f.cb());
}
