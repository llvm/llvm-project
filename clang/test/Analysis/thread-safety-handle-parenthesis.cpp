// RUN: %clang_cc1 -verify -fsyntax-only -std=c++20 -Wthread-safety %s

class __attribute__((lockable)) Lock {};

void sink_protected(int) {}

class Baz {
public:
    Lock lock_;
    int protected_num_ __attribute__((guarded_by(lock_))) = 1;
};

void baz_paran_test() {
    Baz baz;     
    int& n = baz.protected_num_;
    sink_protected(n); // expected-warning{{reading variable 'protected_num_' requires holding mutex 'baz.lock_'}}
    int& n2 = (baz.protected_num_);
    sink_protected(n2); // expected-warning{{reading variable 'protected_num_' requires holding mutex 'baz.lock_'}}
}
