#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Runtime/entry-names.h"
#include "llvm/Support/DebugLog.h"

namespace hlfir {
#define GEN_PASS_DEF_EXPRESSIONSIMPLIFICATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "expression-simplification"

#define INDENT(n) std::string((n) * 2, ' ')

static void removeOperands(mlir::Operation *op, int nestLevel);

static void removeOp(mlir::Operation *op, int parentUses, int nestLevel) {
  int opUses = std::distance(op->getUses().begin(), op->getUses().end());

  if (opUses <= parentUses) {
    LDBG() << INDENT(nestLevel) << "remove: " << *op;
    removeOperands(op, nestLevel);
    op->dropAllReferences();
    op->dropAllUses();
    op->erase();
  }
}

static void removeOp(mlir::Operation *op) {
  removeOp(op, /*parentUses=*/0, /*nestLevel=*/0);
  LDBG();
}

static void removeOperands(mlir::Operation *op, int nestLevel) {
  for (mlir::Value operand : op->getOperands()) {
    if (!operand)
      // Already removed.
      continue;
    if (mlir::Operation *operandOp = operand.getDefiningOp()) {
      int uses = 0;
      for (mlir::Operation *user : operandOp->getUsers())
        if (user == op)
          ++uses;
      removeOp(operandOp, uses, nestLevel + 1);
    }
  }
}

template <typename UserOp>
static UserOp getFirstUser(mlir::Operation *op) {
  auto it = op->user_begin(), end = op->user_end(), prev = it;
  for (; it != end; prev = it++)
    ;
  if (prev != end)
    if (auto userOp = mlir::dyn_cast<UserOp>(*prev))
      return userOp;
  return {};
}

template <typename UserOp>
static UserOp getLastUser(mlir::Operation *op) {
  if (!op->getUsers().empty())
    if (auto userOp = mlir::dyn_cast<UserOp>(*op->user_begin()))
      return userOp;
  return {};
}

template <typename UserOp>
static UserOp getPreviousUser(mlir::Operation *op, mlir::Operation *curUser) {
  for (auto user = op->user_begin(), end = op->user_end(); user != end;
       ++user) {
    if (*user == curUser) {
      if (++user == end)
        break;
      if (auto userOp = mlir::dyn_cast<UserOp>(*user))
        return userOp;
      break;
    }
  }
  return {};
}

// Check if operation has the expected number of uses.
static bool expectUses(mlir::Operation *op, int expUses) {
  int uses = std::distance(op->use_begin(), op->use_end());
  if (uses != expUses) {
    LDBG() << "expectUses: expected " << expUses << ", got " << uses;
    for (mlir::Operation *user : op->getUsers())
      LDBG() << "\t" << *user;
  }
  return uses == expUses;
}

template <typename Op>
static Op expectOp(mlir::Value val) {
  if (Op op = mlir::dyn_cast<Op>(val.getDefiningOp())) {
    LDBG() << op;
    return op;
  }
  return {};
}

static mlir::Value findBoxDef(mlir::Value val) {
  if (auto op = expectOp<fir::ConvertOp>(val)) {
    assert(op->getOperands().size() != 0);
    if (auto boxOp = expectOp<fir::EmboxOp>(op->getOperand(0)))
      return boxOp.getResult();
  }
  return {};
}

namespace {

// This class analyzes a trimmed character and removes the call to trim() (and
// its dependencies) if its result is not used elsewhere.
class TrimRemover {
public:
  TrimRemover(fir::FirOpBuilder &builder, mlir::Value charVal,
              mlir::Value charLenVal)
      : builder(builder), charVal(charVal), charLenVal(charLenVal) {}
  TrimRemover(const TrimRemover &) = delete;

  bool charWasTrimmed();
  void removeTrim();

private:
  // Class inputs.
  fir::FirOpBuilder &builder;
  mlir::Value charVal;
  mlir::Value charLenVal;
  // Operations found while analyzing inputs, that are needed when removing
  // the trim call.
  hlfir::DeclareOp charDeclOp;      // Trim input character.
  fir::CallOp trimCallOp;           // Trim call.
  hlfir::EndAssociateOp endAssocOp; // Trim result association.
  hlfir::DestroyOp destroyExprOp;   // Trim result expression.
  fir::AllocaOp allocaOp;           // Trim result alloca.
};

bool TrimRemover::charWasTrimmed() {
  LDBG() << "\ncharWasTrimmed: " << charVal;

  // Get the declare and expression operations associated to `charVal`, that
  // correspond to the result of trim.
  auto charCvtOp = expectOp<fir::ConvertOp>(charVal);
  auto charLenCvtOp = expectOp<fir::ConvertOp>(charLenVal);
  if (!charCvtOp || !charLenCvtOp || !expectUses(charCvtOp, 1) ||
      !expectUses(charLenCvtOp, 1))
    return false;
  auto assocOp = expectOp<hlfir::AssociateOp>(charCvtOp.getOperand());
  if (!assocOp || !expectUses(assocOp, 3)) // end_associate uses assocOp twice
    return false;
  endAssocOp = getLastUser<hlfir::EndAssociateOp>(assocOp);
  if (!endAssocOp)
    return false;
  auto asExprOp = expectOp<hlfir::AsExprOp>(assocOp.getOperand(0));
  if (!asExprOp || !expectUses(asExprOp, 2))
    return false;
  destroyExprOp = getLastUser<hlfir::DestroyOp>(asExprOp);
  if (!destroyExprOp)
    return false;
  auto declOp = expectOp<hlfir::DeclareOp>(asExprOp.getOperand(0));
  if (!declOp || !expectUses(declOp, 1))
    return false;

  // Get associated box and alloca.
  auto boxAddrOp = expectOp<fir::BoxAddrOp>(declOp.getMemref());
  if (!boxAddrOp || !expectUses(boxAddrOp, 1))
    return false;
  auto loadOp = expectOp<fir::LoadOp>(boxAddrOp.getOperand());
  if (!loadOp || !getFirstUser<fir::BoxEleSizeOp>(loadOp) ||
      !expectUses(loadOp, 2))
    return false;
  // The allocaOp is initialized by a store.
  // Besides its use by the store and loadOp, it's also converted and used by
  // the trim call.
  allocaOp = expectOp<fir::AllocaOp>(loadOp.getMemref());
  if (!allocaOp || !getFirstUser<fir::StoreOp>(allocaOp) ||
      !expectUses(allocaOp, 3))
    return false;

  // Find the trim call that uses the allocaOp.
  if (auto userOp = getPreviousUser<fir::ConvertOp>(allocaOp, loadOp))
    if (userOp->hasOneUse())
      trimCallOp = mlir::dyn_cast<fir::CallOp>(*userOp->user_begin());
  if (!trimCallOp)
    return false;
  LDBG() << "call: " << trimCallOp;
  llvm::StringRef calleeName =
      trimCallOp.getCalleeAttr().getLeafReference().getValue();
  LDBG() << "callee: " << calleeName;
  if (calleeName != RTNAME_STRING(Trim))
    return false;

  // Get trim input character.
  auto chrEmboxOp =
      expectOp<fir::EmboxOp>(findBoxDef(trimCallOp.getOperand(1)));
  if (!chrEmboxOp)
    return false;
  charDeclOp = expectOp<hlfir::DeclareOp>(chrEmboxOp.getMemref());
  if (!charDeclOp)
    return false;

  // Found everything as expected.
  return true;
}

void TrimRemover::removeTrim() {
  LDBG() << "\nremoveTrim:";

  auto charCvtOp = expectOp<fir::ConvertOp>(charVal);
  auto charLenCvtOp = expectOp<fir::ConvertOp>(charLenVal);
  assert(charCvtOp && charLenCvtOp);

  // Replace trim output char with its input.
  mlir::Location loc = charVal.getLoc();
  auto cvtOp = fir::ConvertOp::create(builder, loc, charCvtOp.getType(),
                                      charDeclOp.getOriginalBase());
  charCvtOp.replaceAllUsesWith(cvtOp.getResult());

  // Replace trim output length with its input.
  mlir::Value chrLen = charDeclOp.getTypeparams().back();
  auto cvtLenOp =
      fir::ConvertOp::create(builder, loc, charLenCvtOp.getType(), chrLen);
  charLenCvtOp.replaceAllUsesWith(cvtLenOp.getResult());

  // Remove trim call and old conversions.
  removeOp(charCvtOp);
  removeOp(charLenCvtOp);
  removeOp(trimCallOp);
  // Remove association and expression.
  removeOp(endAssocOp);
  removeOp(destroyExprOp);
  // The only remaining use of allocaOp should be its initialization.
  // Remove the store and alloca operations.
  if (auto userOp = getLastUser<fir::StoreOp>(allocaOp))
    removeOp(userOp);
}

} // namespace

namespace {

class ExpressionSimplification
    : public hlfir::impl::ExpressionSimplificationBase<
          ExpressionSimplification> {
public:
  using ExpressionSimplificationBase<
      ExpressionSimplification>::ExpressionSimplificationBase;

  void runOnOperation() override;

private:
  // Simplify character comparisons.
  // Because character comparison appends spaces to the shorter character,
  // calls to trim() that are used only in the comparison can be eliminated.
  //
  // Example:
  // `trim(x) == trim(y)`
  // can be simplified to
  // `x == y`
  void simplifyCharCompare(fir::CallOp call, const fir::KindMapping &kindMap);
};

void ExpressionSimplification::simplifyCharCompare(
    fir::CallOp call, const fir::KindMapping &kindMap) {
  fir::FirOpBuilder builder{call, kindMap};
  mlir::Operation::operand_range args = call.getArgs();
  TrimRemover lhsTrimRem(builder, args[0], args[2]);
  TrimRemover rhsTrimRem(builder, args[1], args[3]);

  if (lhsTrimRem.charWasTrimmed())
    lhsTrimRem.removeTrim();
  if (rhsTrimRem.charWasTrimmed())
    rhsTrimRem.removeTrim();
}

void ExpressionSimplification::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  fir::KindMapping kindMap = fir::getKindMapping(module);
  module.walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      if (mlir::SymbolRefAttr callee = call.getCalleeAttr()) {
        mlir::StringRef funcName = callee.getLeafReference().getValue();
        if (funcName.starts_with(RTNAME_STRING(CharacterCompareScalar))) {
          simplifyCharCompare(call, kindMap);
        }
      }
    }
  });
}

} // namespace
