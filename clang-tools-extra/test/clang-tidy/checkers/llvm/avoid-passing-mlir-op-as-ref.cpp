// RUN: %check_clang_tidy %s llvm-avoid-passing-mlir-op-as-ref %t

namespace mlir {
template <typename ConcreteType, template <typename T> class... Traits>
class Op {
public:
  // Minimal definition to satisfy isSameOrDerivedFrom
};
} // namespace mlir

class MyOp : public mlir::Op<MyOp> {
  using Op::Op;
};
class OtherClass {};

// Should trigger warning
void badFunction(const MyOp &op) {
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: MLIR Op class 'MyOp' should be passed by value, not by reference [llvm-avoid-passing-mlir-op-as-ref]
}

// Should trigger warning logic for non-const ref too
void badFunctionMutable(MyOp &op) {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: MLIR Op class 'MyOp' should be passed by value, not by reference [llvm-avoid-passing-mlir-op-as-ref]
}

// Good: passed by value
void goodFunction(MyOp op) {}

// Good: not an Op
void otherFunction(const OtherClass &c) {}

// Good: pointer to Op (not common nor necessarily good practice)
void pointerFunction(MyOp *op) {}
