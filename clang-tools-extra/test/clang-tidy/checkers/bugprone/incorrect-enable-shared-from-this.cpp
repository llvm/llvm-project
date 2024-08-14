// RUN: %check_clang_tidy %s bugprone-incorrect-enable-shared-from-this %t -- --

namespace std {
    template <typename T> class enable_shared_from_this {};
} //namespace std

class BadClassExample : std::enable_shared_from_this<BadClassExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadClassExample>

class BadClass2Example : private std::enable_shared_from_this<BadClass2Example> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadClass2Example>

struct BadStructExample : private std::enable_shared_from_this<BadStructExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadStructExample>

class GoodClassExample : public std::enable_shared_from_this<GoodClassExample> {};

struct GoodStructExample : public std::enable_shared_from_this<GoodStructExample> {};

struct GoodStruct2Example : std::enable_shared_from_this<GoodStruct2Example> {};

class dummy_class1 {};
class dummy_class2 {};

class BadMultiClassExample : std::enable_shared_from_this<BadMultiClassExample>, dummy_class1 {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadMultiClassExample>, dummy_class1

class BadMultiClass2Example : dummy_class1, std::enable_shared_from_this<BadMultiClass2Example>, dummy_class2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: dummy_class1, public std::enable_shared_from_this<BadMultiClass2Example>, dummy_class2

class BadMultiClass3Example : dummy_class1, dummy_class2, std::enable_shared_from_this<BadMultiClass3Example> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: dummy_class1, dummy_class2, public std::enable_shared_from_this<BadMultiClass3Example>

template <typename T> class enable_shared_from_this {};

class BadInitClassExample : public enable_shared_from_this<BadInitClassExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Should be std::enable_shared_from_this [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadInitClassExample>

struct BadInitStructExample : enable_shared_from_this<BadInitStructExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: Should be std::enable_shared_from_this [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: std::enable_shared_from_this<BadInitStructExample>

class BadMixedProblemExample : enable_shared_from_this<BadMixedProblemExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Should be std::enable_shared_from_this and inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: public std::enable_shared_from_this<BadMixedProblemExample>

class ClassBase : public std::enable_shared_from_this<ClassBase> {};
class PrivateInheritClassBase : private ClassBase{};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class PrivateInheritClassBase : public ClassBase{};

class DefaultInheritClassBase : ClassBase{};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class DefaultInheritClassBase : public ClassBase{};

class PublicInheritClassBase : public ClassBase{};

struct StructBase : public std::enable_shared_from_this<StructBase> {};
struct PrivateInheritStructBase : private StructBase{};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: inheritance from std::enable_shared_from_this should be public inheritance, otherwise the internal weak_ptr won't be initialized [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: struct PrivateInheritStructBase : public StructBase{};

struct DefaultInheritStructBase : StructBase{};

struct PublicInheritStructBase : StructBase{};
