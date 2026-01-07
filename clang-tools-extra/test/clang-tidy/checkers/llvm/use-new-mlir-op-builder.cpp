// RUN: %check_clang_tidy %s llvm-use-new-mlir-op-builder %t

namespace mlir {
class Location {};
class Value {};
class OpBuilder {
public:
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return OpTy(args...);
  }
  Location getUnknownLoc() { return Location(); }
};
class ImplicitLocOpBuilder : public OpBuilder {
public:
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return OpTy(args...);
  }
};
struct ModuleOp {
  ModuleOp() {}
  static ModuleOp create(OpBuilder &builder, Location location) {
    return ModuleOp();
  }
};
struct NamedOp {
  NamedOp(const char* name) {}
  static NamedOp create(OpBuilder &builder, Location location, const char* name) {
    return NamedOp(name);
  }
  Value getResult() { return Value(); }
};
struct OperandOp {
  OperandOp(Value val) {}
  static OperandOp create(OpBuilder &builder, Location location, Value val) {
    return OperandOp(val);
  }
};
} // namespace mlir

#define ASSIGN(A, B, C, D) C A = B.create<C>(B.getUnknownLoc(), D)

template <typename T>
void g(mlir::OpBuilder &b) {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: T::create(b, b.getUnknownLoc(), "gaz");
  b.create<T>(b.getUnknownLoc(), "gaz");
}

class CustomBuilder : public mlir::ImplicitLocOpBuilder {
public:
  mlir::NamedOp f(const char *name) {
    // CHECK-MESSAGES: :[[@LINE+2]]:12: warning: use 'OpType::create(builder, ...)'
    // CHECK-FIXES: return mlir::NamedOp::create(*this, name);
    return create<mlir::NamedOp>(name);
  }

  mlir::NamedOp g(const char *name) {
    using mlir::NamedOp;
    // CHECK-MESSAGES: :[[@LINE+2]]:12: warning: use 'OpType::create(builder, ...)'
    // CHECK-FIXES: return NamedOp::create(*this, name);
    return create<NamedOp>(name);
  }
};

void f() {
  mlir::OpBuilder builder;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: mlir::  ModuleOp::create(builder, builder.getUnknownLoc());
  builder.create<mlir::  ModuleOp>(builder.getUnknownLoc());

  using mlir::NamedOp;
  using mlir::OperandOp;

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: NamedOp::create(builder, builder.getUnknownLoc(), "baz");
  builder.create<NamedOp>(builder.getUnknownLoc(), "baz");

  // CHECK-MESSAGES: :[[@LINE+4]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: NamedOp::create(builder,
  // CHECK-FIXES:      builder.getUnknownLoc(),
  // CHECK-FIXES:      "caz");
  builder.
   create<NamedOp>  (
     builder.getUnknownLoc(),
     "caz");

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  ASSIGN(op, builder, NamedOp, "daz");

  g<NamedOp>(builder);

  mlir::ImplicitLocOpBuilder ib;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: mlir::ModuleOp::create(ib );
  ib.create<mlir::ModuleOp>(   );

  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-FIXES: mlir::OpBuilder().create<mlir::ModuleOp>(builder.getUnknownLoc());
  mlir::OpBuilder().create<mlir::ModuleOp>(builder.getUnknownLoc());

  auto *p = &builder;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)'
  // CHECK-FIXES: NamedOp::create(*p, builder.getUnknownLoc(), "eaz");
  p->create<NamedOp>(builder.getUnknownLoc(), "eaz");

  CustomBuilder cb;
  cb.f("faz");
  cb.g("gaz");

  // CHECK-FIXES:      OperandOp::create(builder, builder.getUnknownLoc(),
  // CHECK-FIXES-NEXT:   NamedOp::create(builder, builder.getUnknownLoc(), "haz").getResult());
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  // CHECK-MESSAGES: :[[@LINE+2]]:5: warning: use 'OpType::create(builder, ...)' instead of 'builder.create<OpType>(...)' [llvm-use-new-mlir-op-builder]
  builder.create<OperandOp>(builder.getUnknownLoc(),
    builder.create<NamedOp>(builder.getUnknownLoc(), "haz").getResult());
}
