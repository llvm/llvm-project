// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char*, ...);

// Test 1: Pointer to private type alias (the original bug report)
class io_context { using impl_type = int; public: impl_type* get(); };
io_context::impl_type* io_context::get() { return nullptr; }
printf("Pointer to private type: %s\n", io_context().get() == nullptr ? "passed" : "failed");
// CHECK: Pointer to private type: passed

// Test 2: Reference to private type
class RefReturn { using ref_t = int; ref_t value = 42; public: ref_t& getRef(); };
RefReturn::ref_t& RefReturn::getRef() { return value; }
printf("Reference to private type: %d\n", RefReturn().getRef());
// CHECK: Reference to private type: 42

// Test 3: Double pointer to private type
class PtrPtr { using inner_t = int; public: inner_t** get(); };
PtrPtr::inner_t** PtrPtr::get() { static int* p = nullptr; return &p; }
printf("Double pointer to private type: %s\n", PtrPtr().get() != nullptr ? "passed" : "failed");
// CHECK: Double pointer to private type: passed

// Test 4: Const reference to private type
class ConstRef { using data_t = int; data_t val = 100; public: const data_t& get(); };
const ConstRef::data_t& ConstRef::get() { return val; }
printf("Const reference to private type: %d\n", ConstRef().get());
// CHECK: Const reference to private type: 100

// Test 5: Pointer to private nested struct
class Container { struct Node { int x; }; public: Node* create(); };
Container::Node* Container::create() { return new Node{789}; }
printf("Pointer to private nested struct: %d\n", Container().create()->x);
// CHECK: Pointer to private nested struct: 789

%quit
