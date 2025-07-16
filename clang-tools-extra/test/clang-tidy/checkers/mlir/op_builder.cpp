// RUN: %check_clang_tidy --match-partial-fixes %s mlir-op-builder %t

namespace mlir {
class Location {};
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
};
} // namespace mlir

void f() {
  mlir::OpBuilder builder;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: Use OpType::create(builder, ...) instead of builder.create<OpType>(...) [mlir-op-builder]
  // CHECK-FIXES: mlir::  ModuleOp::create(builder, builder.getUnknownLoc())
  builder.create<mlir::  ModuleOp>(builder.getUnknownLoc());

  using mlir::NamedOp;
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: Use OpType::create(builder, ...) instead of builder.create<OpType>(...) [mlir-op-builder]
  // CHECK-FIXES: NamedOp::create(builder, builder.getUnknownLoc(), "baz")
  builder.create<NamedOp>(builder.getUnknownLoc(), "baz");

  mlir::ImplicitLocOpBuilder ib;
  // Note: extra space in the case where there is no other arguments. Could be
  // improved, but also clang-format will do that just post.
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: Use OpType::create(builder, ...) instead of builder.create<OpType>(...) [mlir-op-builder]
  // CHECK-FIXES: mlir::ModuleOp::create(ib )
  ib.create<mlir::ModuleOp>();
}
