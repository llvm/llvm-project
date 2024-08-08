// RUN: %check_clang_tidy %s bugprone-incorrect-enable-shared-from-this %t -- --

namespace std {

    template <typename T> class enable_shared_from_this {};

}

class BadClassExample : std::enable_shared_from_this<BadClassExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class BadExample is not public even though it's derived from std::enable_shared_from_this [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public enable_shared_from_this<BadExample>

class BadClass2Example : private std::enable_shared_from_this<BadClass2Example> {};

struct BadStructExample : private std::enable_shared_from_this<BadStructExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class BadStructExample is not public even though it's derived from std::enable_shared_from_this [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadStructExample>

class GoodClassExample : public std::enable_shared_from_this<GoodClassExample> {};

struct GoodStructExample : public std::enable_shared_from_this<GoodStructExample> {};

struct GoodStruct2Example : std::enable_shared_from_this<GoodStruct2Example> {};
