// RUN: %check_clang_tidy %s bugprone-public-enable-shared-from-this %t -- --

namespace std {

    template <typename T> class enable_shared_from_this {};

    class BadExample : enable_shared_from_this<BadExample> {};
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class BadExample is not public even though it's derived from std::enable_shared_from_this [bugprone-public-enable-shared-from-this]
    // CHECK-FIXES: public enable_shared_from_this<BadExample>

    class Bad2Example : std::enable_shared_from_this<Bad2Example> {};
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class Bad2Example is not public even though it's derived from std::enable_shared_from_this [bugprone-public-enable-shared-from-this]
    // CHECK-FIXES: public std::enable_shared_from_this<Bad2Example>

    class GoodExample : public enable_shared_from_this<GoodExample> {
    };

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
}