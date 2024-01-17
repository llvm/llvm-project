namespace n {
    namespace m {
        class C {
        public:
            void f() {}
            char* g(int);
        };
    }
}

void globalF() {
    n::m::C c;
    c.f();
    c.g(0);
}