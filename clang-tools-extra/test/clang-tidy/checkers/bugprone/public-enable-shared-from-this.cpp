// RUN: %check_clang_tidy %s bugprone-public-enable-shared-from-this %t -- -- -I %S/Inputs/
#include <memory>

class BadExample : std::enable_shared_from_this<BadExample> {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class BadExample is not public even though it's derived from std::enable_shared_from_this [bugprone-public-enable-shared-from-this]
// CHECK-FIXES: :[[@LINE-2]]:19 public 
        public:
        BadExample* foo() { return shared_from_this().get(); }
        void bar() { return; }
};

void using_not_public() {
        auto bad_example = std::make_shared<BadExample>();
        auto* b_ex = bad_example->foo();
        b_ex->bar();
}

class GoodExample : public std::enable_shared_from_this<GoodExample> {
        public:
        GoodExample* foo() { return shared_from_this().get(); }
        void bar() { return; }
};

void using_public() {
        auto good_example = std::make_shared<GoodExample>();
        auto* g_ex = good_example->foo();
        g_ex->bar();
}

struct BaseClass {

    void print() {
        (void) State;
        (void) Requester;
    }
    bool State;
    int Requester;
};

class InheritPrivateBaseClass : BaseClass {
    public:
        void additionalFunction() {
            (void) ID;
        }
    private: 
        int ID;
};

class InheritPublicBaseClass : public BaseClass {
    public:
        void additionalFunction() {
            (void) ID;
        }
    private: 
        int ID;
};