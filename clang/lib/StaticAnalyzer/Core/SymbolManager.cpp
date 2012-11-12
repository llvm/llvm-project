//== SymbolManager.h - Management of Symbolic Values ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolManager, a class that manages symbolic values
//  created for use by ExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

void SymExpr::anchor() { }

void SymExpr::dump() const {
  dumpToStream(llvm::errs());
}

static void print(raw_ostream &os, BinaryOperator::Opcode Op) {
  switch (Op) {
    default:
      llvm_unreachable("operator printing not implemented");
    case BO_Mul: os << '*'  ; break;
    case BO_Div: os << '/'  ; break;
    case BO_Rem: os << '%'  ; break;
    case BO_Add: os << '+'  ; break;
    case BO_Sub: os << '-'  ; break;
    case BO_Shl: os << "<<" ; break;
    case BO_Shr: os << ">>" ; break;
    case BO_LT:  os << "<"  ; break;
    case BO_GT:  os << '>'  ; break;
    case BO_LE:  os << "<=" ; break;
    case BO_GE:  os << ">=" ; break;
    case BO_EQ:  os << "==" ; break;
    case BO_NE:  os << "!=" ; break;
    case BO_And: os << '&'  ; break;
    case BO_Xor: os << '^'  ; break;
    case BO_Or:  os << '|'  ; break;
  }
}

void SymIntExpr::dumpToStream(raw_ostream &os) const {
  os << '(';
  getLHS()->dumpToStream(os);
  os << ") ";
  print(os, getOpcode());
  os << ' ' << getRHS().getZExtValue();
  if (getRHS().isUnsigned()) os << 'U';
}

void IntSymExpr::dumpToStream(raw_ostream &os) const {
  os << ' ' << getLHS().getZExtValue();
  if (getLHS().isUnsigned()) os << 'U';
  print(os, getOpcode());
  os << '(';
  getRHS()->dumpToStream(os);
  os << ") ";
}

void SymSymExpr::dumpToStream(raw_ostream &os) const {
  os << '(';
  getLHS()->dumpToStream(os);
  os << ") ";
  os << '(';
  getRHS()->dumpToStream(os);
  os << ')';
}

void SymbolCast::dumpToStream(raw_ostream &os) const {
  os << '(' << ToTy.getAsString() << ") (";
  Operand->dumpToStream(os);
  os << ')';
}

void SymbolConjured::dumpToStream(raw_ostream &os) const {
  os << "conj_$" << getSymbolID() << '{' << T.getAsString() << '}';
}

void SymbolDerived::dumpToStream(raw_ostream &os) const {
  os << "derived_$" << getSymbolID() << '{'
     << getParentSymbol() << ',' << getRegion() << '}';
}

void SymbolExtent::dumpToStream(raw_ostream &os) const {
  os << "extent_$" << getSymbolID() << '{' << getRegion() << '}';
}

void SymbolMetadata::dumpToStream(raw_ostream &os) const {
  os << "meta_$" << getSymbolID() << '{'
     << getRegion() << ',' << T.getAsString() << '}';
}

void SymbolData::anchor() { }

void SymbolRegionValue::dumpToStream(raw_ostream &os) const {
  os << "reg_$" << getSymbolID() << "<" << R << ">";
}

bool SymExpr::symbol_iterator::operator==(const symbol_iterator &X) const {
  return itr == X.itr;
}

bool SymExpr::symbol_iterator::operator!=(const symbol_iterator &X) const {
  return itr != X.itr;
}

SymExpr::symbol_iterator::symbol_iterator(const SymExpr *SE) {
  itr.push_back(SE);
}

SymExpr::symbol_iterator &SymExpr::symbol_iterator::operator++() {
  assert(!itr.empty() && "attempting to iterate on an 'end' iterator");
  expand();
  return *this;
}

SymbolRef SymExpr::symbol_iterator::operator*() {
  assert(!itr.empty() && "attempting to dereference an 'end' iterator");
  return itr.back();
}

void SymExpr::symbol_iterator::expand() {
  const SymExpr *SE = itr.back();
  itr.pop_back();

  switch (SE->getKind()) {
    case SymExpr::RegionValueKind:
    case SymExpr::ConjuredKind:
    case SymExpr::DerivedKind:
    case SymExpr::ExtentKind:
    case SymExpr::MetadataKind:
      return;
    case SymExpr::CastSymbolKind:
      itr.push_back(cast<SymbolCast>(SE)->getOperand());
      return;
    case SymExpr::SymIntKind:
      itr.push_back(cast<SymIntExpr>(SE)->getLHS());
      return;
    case SymExpr::IntSymKind:
      itr.push_back(cast<IntSymExpr>(SE)->getRHS());
      return;
    case SymExpr::SymSymKind: {
      const SymSymExpr *x = cast<SymSymExpr>(SE);
      itr.push_back(x->getLHS());
      itr.push_back(x->getRHS());
      return;
    }
  }
  llvm_unreachable("unhandled expansion case");
}

unsigned SymExpr::computeComplexity() const {
  unsigned R = 0;
  for (symbol_iterator I = symbol_begin(), E = symbol_end(); I != E; ++I)
    R++;
  return R;
}

const SymbolRegionValue*
SymbolManager::getRegionValueSymbol(const TypedValueRegion* R) {
  llvm::FoldingSetNodeID profile;
  SymbolRegionValue::Profile(profile, R);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolRegionValue>();
    new (SD) SymbolRegionValue(SymbolCounter, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolRegionValue>(SD);
}

const SymbolConjured* SymbolManager::conjureSymbol(const Stmt *E,
                                                   const LocationContext *LCtx,
                                                   QualType T,
                                                   unsigned Count,
                                                   const void *SymbolTag) {
  llvm::FoldingSetNodeID profile;
  SymbolConjured::Profile(profile, E, T, Count, LCtx, SymbolTag);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolConjured>();
    new (SD) SymbolConjured(SymbolCounter, E, LCtx, T, Count, SymbolTag);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolConjured>(SD);
}

const SymbolDerived*
SymbolManager::getDerivedSymbol(SymbolRef parentSymbol,
                                const TypedValueRegion *R) {

  llvm::FoldingSetNodeID profile;
  SymbolDerived::Profile(profile, parentSymbol, R);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolDerived>();
    new (SD) SymbolDerived(SymbolCounter, parentSymbol, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolDerived>(SD);
}

const SymbolExtent*
SymbolManager::getExtentSymbol(const SubRegion *R) {
  llvm::FoldingSetNodeID profile;
  SymbolExtent::Profile(profile, R);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolExtent>();
    new (SD) SymbolExtent(SymbolCounter, R);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolExtent>(SD);
}

const SymbolMetadata*
SymbolManager::getMetadataSymbol(const MemRegion* R, const Stmt *S, QualType T,
                                 unsigned Count, const void *SymbolTag) {

  llvm::FoldingSetNodeID profile;
  SymbolMetadata::Profile(profile, R, S, T, Count, SymbolTag);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = (SymExpr*) BPAlloc.Allocate<SymbolMetadata>();
    new (SD) SymbolMetadata(SymbolCounter, R, S, T, Count, SymbolTag);
    DataSet.InsertNode(SD, InsertPos);
    ++SymbolCounter;
  }

  return cast<SymbolMetadata>(SD);
}

const SymbolCast*
SymbolManager::getCastSymbol(const SymExpr *Op,
                             QualType From, QualType To) {
  llvm::FoldingSetNodeID ID;
  SymbolCast::Profile(ID, Op, From, To);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);
  if (!data) {
    data = (SymbolCast*) BPAlloc.Allocate<SymbolCast>();
    new (data) SymbolCast(Op, From, To);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<SymbolCast>(data);
}

const SymIntExpr *SymbolManager::getSymIntExpr(const SymExpr *lhs,
                                               BinaryOperator::Opcode op,
                                               const llvm::APSInt& v,
                                               QualType t) {
  llvm::FoldingSetNodeID ID;
  SymIntExpr::Profile(ID, lhs, op, v, t);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);

  if (!data) {
    data = (SymIntExpr*) BPAlloc.Allocate<SymIntExpr>();
    new (data) SymIntExpr(lhs, op, v, t);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<SymIntExpr>(data);
}

const IntSymExpr *SymbolManager::getIntSymExpr(const llvm::APSInt& lhs,
                                               BinaryOperator::Opcode op,
                                               const SymExpr *rhs,
                                               QualType t) {
  llvm::FoldingSetNodeID ID;
  IntSymExpr::Profile(ID, lhs, op, rhs, t);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);

  if (!data) {
    data = (IntSymExpr*) BPAlloc.Allocate<IntSymExpr>();
    new (data) IntSymExpr(lhs, op, rhs, t);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<IntSymExpr>(data);
}

const SymSymExpr *SymbolManager::getSymSymExpr(const SymExpr *lhs,
                                               BinaryOperator::Opcode op,
                                               const SymExpr *rhs,
                                               QualType t) {
  llvm::FoldingSetNodeID ID;
  SymSymExpr::Profile(ID, lhs, op, rhs, t);
  void *InsertPos;
  SymExpr *data = DataSet.FindNodeOrInsertPos(ID, InsertPos);

  if (!data) {
    data = (SymSymExpr*) BPAlloc.Allocate<SymSymExpr>();
    new (data) SymSymExpr(lhs, op, rhs, t);
    DataSet.InsertNode(data, InsertPos);
  }

  return cast<SymSymExpr>(data);
}

QualType SymbolConjured::getType() const {
  return T;
}

QualType SymbolDerived::getType() const {
  return R->getValueType();
}

QualType SymbolExtent::getType() const {
  ASTContext &Ctx = R->getMemRegionManager()->getContext();
  return Ctx.getSizeType();
}

QualType SymbolMetadata::getType() const {
  return T;
}

QualType SymbolRegionValue::getType() const {
  return R->getValueType();
}

SymbolManager::~SymbolManager() {
  for (SymbolDependTy::const_iterator I = SymbolDependencies.begin(),
       E = SymbolDependencies.end(); I != E; ++I) {
    delete I->second;
  }

}

bool SymbolManager::canSymbolicate(QualType T) {
  T = T.getCanonicalType();

  if (Loc::isLocType(T))
    return true;

  if (T->isIntegerType())
    return T->isScalarType();

  if (T->isRecordType() && !T->isUnionType())
    return true;

  return false;
}

void SymbolManager::addSymbolDependency(const SymbolRef Primary,
                                        const SymbolRef Dependent) {
  SymbolDependTy::iterator I = SymbolDependencies.find(Primary);
  SymbolRefSmallVectorTy *dependencies = 0;
  if (I == SymbolDependencies.end()) {
    dependencies = new SymbolRefSmallVectorTy();
    SymbolDependencies[Primary] = dependencies;
  } else {
    dependencies = I->second;
  }
  dependencies->push_back(Dependent);
}

const SymbolRefSmallVectorTy *SymbolManager::getDependentSymbols(
                                                     const SymbolRef Primary) {
  SymbolDependTy::const_iterator I = SymbolDependencies.find(Primary);
  if (I == SymbolDependencies.end())
    return 0;
  return I->second;
}

void SymbolReaper::markDependentsLive(SymbolRef sym) {
  // Do not mark dependents more then once.
  SymbolMapTy::iterator LI = TheLiving.find(sym);
  assert(LI != TheLiving.end() && "The primary symbol is not live.");
  if (LI->second == HaveMarkedDependents)
    return;
  LI->second = HaveMarkedDependents;

  if (const SymbolRefSmallVectorTy *Deps = SymMgr.getDependentSymbols(sym)) {
    for (SymbolRefSmallVectorTy::const_iterator I = Deps->begin(),
                                                E = Deps->end(); I != E; ++I) {
      if (TheLiving.find(*I) != TheLiving.end())
        continue;
      markLive(*I);
    }
  }
}

void SymbolReaper::markLive(SymbolRef sym) {
  TheLiving[sym] = NotProcessed;
  TheDead.erase(sym);
  markDependentsLive(sym);
}

void SymbolReaper::markLive(const MemRegion *region) {
  RegionRoots.insert(region);
}

void SymbolReaper::markInUse(SymbolRef sym) {
  if (isa<SymbolMetadata>(sym))
    MetadataInUse.insert(sym);
}

bool SymbolReaper::maybeDead(SymbolRef sym) {
  if (isLive(sym))
    return false;

  TheDead.insert(sym);
  return true;
}

bool SymbolReaper::isLiveRegion(const MemRegion *MR) {
  if (RegionRoots.count(MR))
    return true;
  
  MR = MR->getBaseRegion();

  if (const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(MR))
    return isLive(SR->getSymbol());

  if (const VarRegion *VR = dyn_cast<VarRegion>(MR))
    return isLive(VR, true);

  // FIXME: This is a gross over-approximation. What we really need is a way to
  // tell if anything still refers to this region. Unlike SymbolicRegions,
  // AllocaRegions don't have associated symbols, though, so we don't actually
  // have a way to track their liveness.
  if (isa<AllocaRegion>(MR))
    return true;

  if (isa<CXXThisRegion>(MR))
    return true;

  if (isa<MemSpaceRegion>(MR))
    return true;

  return false;
}

bool SymbolReaper::isLive(SymbolRef sym) {
  if (TheLiving.count(sym)) {
    markDependentsLive(sym);
    return true;
  }
  
  bool KnownLive;
  
  switch (sym->getKind()) {
  case SymExpr::RegionValueKind:
    // FIXME: We should be able to use isLiveRegion here (this behavior
    // predates isLiveRegion), but doing so causes test failures. Investigate.
    KnownLive = true;
    break;
  case SymExpr::ConjuredKind:
    KnownLive = false;
    break;
  case SymExpr::DerivedKind:
    KnownLive = isLive(cast<SymbolDerived>(sym)->getParentSymbol());
    break;
  case SymExpr::ExtentKind:
    KnownLive = isLiveRegion(cast<SymbolExtent>(sym)->getRegion());
    break;
  case SymExpr::MetadataKind:
    KnownLive = MetadataInUse.count(sym) &&
                isLiveRegion(cast<SymbolMetadata>(sym)->getRegion());
    if (KnownLive)
      MetadataInUse.erase(sym);
    break;
  case SymExpr::SymIntKind:
    KnownLive = isLive(cast<SymIntExpr>(sym)->getLHS());
    break;
  case SymExpr::IntSymKind:
    KnownLive = isLive(cast<IntSymExpr>(sym)->getRHS());
    break;
  case SymExpr::SymSymKind:
    KnownLive = isLive(cast<SymSymExpr>(sym)->getLHS()) &&
                isLive(cast<SymSymExpr>(sym)->getRHS());
    break;
  case SymExpr::CastSymbolKind:
    KnownLive = isLive(cast<SymbolCast>(sym)->getOperand());
    break;
  }

  if (KnownLive)
    markLive(sym);

  return KnownLive;
}

bool
SymbolReaper::isLive(const Stmt *ExprVal, const LocationContext *ELCtx) const {
  if (LCtx == 0)
    return false;

  if (LCtx != ELCtx) {
    // If the reaper's location context is a parent of the expression's
    // location context, then the expression value is now "out of scope".
    if (LCtx->isParentOf(ELCtx))
      return false;
    return true;
  }

  // If no statement is provided, everything is this and parent contexts is live.
  if (!Loc)
    return true;

  return LCtx->getAnalysis<RelaxedLiveVariables>()->isLive(Loc, ExprVal);
}

bool SymbolReaper::isLive(const VarRegion *VR, bool includeStoreBindings) const{
  const StackFrameContext *VarContext = VR->getStackFrame();

  if (!VarContext)
    return true;

  if (!LCtx)
    return false;
  const StackFrameContext *CurrentContext = LCtx->getCurrentStackFrame();

  if (VarContext == CurrentContext) {
    // If no statement is provided, everything is live.
    if (!Loc)
      return true;

    if (LCtx->getAnalysis<RelaxedLiveVariables>()->isLive(Loc, VR->getDecl()))
      return true;

    if (!includeStoreBindings)
      return false;
    
    unsigned &cachedQuery =
      const_cast<SymbolReaper*>(this)->includedRegionCache[VR];

    if (cachedQuery) {
      return cachedQuery == 1;
    }

    // Query the store to see if the region occurs in any live bindings.
    if (Store store = reapedStore.getStore()) {
      bool hasRegion = 
        reapedStore.getStoreManager().includedInBindings(store, VR);
      cachedQuery = hasRegion ? 1 : 2;
      return hasRegion;
    }
    
    return false;
  }

  return VarContext->isParentOf(CurrentContext);
}

SymbolVisitor::~SymbolVisitor() {}
