// RUN: %clang_cc1 -emit-llvm -o - %s

// PR5834
struct ASTMultiMover {};
struct ASTMultiPtr {
  ASTMultiPtr();
  ASTMultiPtr(ASTMultiPtr&);
  ASTMultiPtr(ASTMultiMover mover);
  operator ASTMultiMover();
};
void f1() {
  extern void f0(ASTMultiPtr);
  f0(ASTMultiPtr());
}
