//== Store.cpp - Interface for maps from Locations to Values ----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the types Store and StoreManager.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"

using namespace clang;
using namespace ento;

StoreManager::StoreManager(ProgramStateManager &stateMgr)
  : svalBuilder(stateMgr.getSValBuilder()), StateMgr(stateMgr),
    MRMgr(svalBuilder.getRegionManager()), Ctx(stateMgr.getContext()) {}

StoreRef StoreManager::enterStackFrame(Store OldStore,
                                       const CallEvent &Call,
                                       const StackFrameContext *LCtx) {
  StoreRef Store = StoreRef(OldStore, *this);

  SmallVector<CallEvent::FrameBindingTy, 16> InitialBindings;
  Call.getInitialStackFrameContents(LCtx, InitialBindings);

  for (CallEvent::BindingsTy::iterator I = InitialBindings.begin(),
                                       E = InitialBindings.end();
       I != E; ++I) {
    Store = Bind(Store.getStore(), I->first, I->second);
  }

  return Store;
}

const MemRegion *StoreManager::MakeElementRegion(const MemRegion *Base,
                                              QualType EleTy, uint64_t index) {
  NonLoc idx = svalBuilder.makeArrayIndex(index);
  return MRMgr.getElementRegion(EleTy, idx, Base, svalBuilder.getContext());
}

// FIXME: Merge with the implementation of the same method in MemRegion.cpp
static bool IsCompleteType(ASTContext &Ctx, QualType Ty) {
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *D = RT->getDecl();
    if (!D->getDefinition())
      return false;
  }

  return true;
}

StoreRef StoreManager::BindDefault(Store store, const MemRegion *R, SVal V) {
  return StoreRef(store, *this);
}

const ElementRegion *StoreManager::GetElementZeroRegion(const MemRegion *R, 
                                                        QualType T) {
  NonLoc idx = svalBuilder.makeZeroArrayIndex();
  assert(!T.isNull());
  return MRMgr.getElementRegion(T, idx, R, Ctx);
}

const MemRegion *StoreManager::castRegion(const MemRegion *R, QualType CastToTy) {

  ASTContext &Ctx = StateMgr.getContext();

  // Handle casts to Objective-C objects.
  if (CastToTy->isObjCObjectPointerType())
    return R->StripCasts();

  if (CastToTy->isBlockPointerType()) {
    // FIXME: We may need different solutions, depending on the symbol
    // involved.  Blocks can be casted to/from 'id', as they can be treated
    // as Objective-C objects.  This could possibly be handled by enhancing
    // our reasoning of downcasts of symbolic objects.
    if (isa<CodeTextRegion>(R) || isa<SymbolicRegion>(R))
      return R;

    // We don't know what to make of it.  Return a NULL region, which
    // will be interpretted as UnknownVal.
    return NULL;
  }

  // Now assume we are casting from pointer to pointer. Other cases should
  // already be handled.
  QualType PointeeTy = CastToTy->getPointeeType();
  QualType CanonPointeeTy = Ctx.getCanonicalType(PointeeTy);

  // Handle casts to void*.  We just pass the region through.
  if (CanonPointeeTy.getLocalUnqualifiedType() == Ctx.VoidTy)
    return R;

  // Handle casts from compatible types.
  if (R->isBoundable())
    if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
      QualType ObjTy = Ctx.getCanonicalType(TR->getValueType());
      if (CanonPointeeTy == ObjTy)
        return R;
    }

  // Process region cast according to the kind of the region being cast.
  switch (R->getKind()) {
    case MemRegion::CXXThisRegionKind:
    case MemRegion::GenericMemSpaceRegionKind:
    case MemRegion::StackLocalsSpaceRegionKind:
    case MemRegion::StackArgumentsSpaceRegionKind:
    case MemRegion::HeapSpaceRegionKind:
    case MemRegion::UnknownSpaceRegionKind:
    case MemRegion::StaticGlobalSpaceRegionKind:
    case MemRegion::GlobalInternalSpaceRegionKind:
    case MemRegion::GlobalSystemSpaceRegionKind:
    case MemRegion::GlobalImmutableSpaceRegionKind: {
      llvm_unreachable("Invalid region cast");
    }

    case MemRegion::FunctionTextRegionKind:
    case MemRegion::BlockTextRegionKind:
    case MemRegion::BlockDataRegionKind:
    case MemRegion::StringRegionKind:
      // FIXME: Need to handle arbitrary downcasts.
    case MemRegion::SymbolicRegionKind:
    case MemRegion::AllocaRegionKind:
    case MemRegion::CompoundLiteralRegionKind:
    case MemRegion::FieldRegionKind:
    case MemRegion::ObjCIvarRegionKind:
    case MemRegion::ObjCStringRegionKind:
    case MemRegion::VarRegionKind:
    case MemRegion::CXXTempObjectRegionKind:
    case MemRegion::CXXBaseObjectRegionKind:
      return MakeElementRegion(R, PointeeTy);

    case MemRegion::ElementRegionKind: {
      // If we are casting from an ElementRegion to another type, the
      // algorithm is as follows:
      //
      // (1) Compute the "raw offset" of the ElementRegion from the
      //     base region.  This is done by calling 'getAsRawOffset()'.
      //
      // (2a) If we get a 'RegionRawOffset' after calling
      //      'getAsRawOffset()', determine if the absolute offset
      //      can be exactly divided into chunks of the size of the
      //      casted-pointee type.  If so, create a new ElementRegion with
      //      the pointee-cast type as the new ElementType and the index
      //      being the offset divded by the chunk size.  If not, create
      //      a new ElementRegion at offset 0 off the raw offset region.
      //
      // (2b) If we don't a get a 'RegionRawOffset' after calling
      //      'getAsRawOffset()', it means that we are at offset 0.
      //
      // FIXME: Handle symbolic raw offsets.

      const ElementRegion *elementR = cast<ElementRegion>(R);
      const RegionRawOffset &rawOff = elementR->getAsArrayOffset();
      const MemRegion *baseR = rawOff.getRegion();

      // If we cannot compute a raw offset, throw up our hands and return
      // a NULL MemRegion*.
      if (!baseR)
        return NULL;

      CharUnits off = rawOff.getOffset();

      if (off.isZero()) {
        // Edge case: we are at 0 bytes off the beginning of baseR.  We
        // check to see if type we are casting to is the same as the base
        // region.  If so, just return the base region.
        if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(baseR)) {
          QualType ObjTy = Ctx.getCanonicalType(TR->getValueType());
          QualType CanonPointeeTy = Ctx.getCanonicalType(PointeeTy);
          if (CanonPointeeTy == ObjTy)
            return baseR;
        }

        // Otherwise, create a new ElementRegion at offset 0.
        return MakeElementRegion(baseR, PointeeTy);
      }

      // We have a non-zero offset from the base region.  We want to determine
      // if the offset can be evenly divided by sizeof(PointeeTy).  If so,
      // we create an ElementRegion whose index is that value.  Otherwise, we
      // create two ElementRegions, one that reflects a raw offset and the other
      // that reflects the cast.

      // Compute the index for the new ElementRegion.
      int64_t newIndex = 0;
      const MemRegion *newSuperR = 0;

      // We can only compute sizeof(PointeeTy) if it is a complete type.
      if (IsCompleteType(Ctx, PointeeTy)) {
        // Compute the size in **bytes**.
        CharUnits pointeeTySize = Ctx.getTypeSizeInChars(PointeeTy);
        if (!pointeeTySize.isZero()) {
          // Is the offset a multiple of the size?  If so, we can layer the
          // ElementRegion (with elementType == PointeeTy) directly on top of
          // the base region.
          if (off % pointeeTySize == 0) {
            newIndex = off / pointeeTySize;
            newSuperR = baseR;
          }
        }
      }

      if (!newSuperR) {
        // Create an intermediate ElementRegion to represent the raw byte.
        // This will be the super region of the final ElementRegion.
        newSuperR = MakeElementRegion(baseR, Ctx.CharTy, off.getQuantity());
      }

      return MakeElementRegion(newSuperR, PointeeTy, newIndex);
    }
  }

  llvm_unreachable("unreachable");
}

SVal StoreManager::evalDerivedToBase(SVal Derived, const CastExpr *Cast) {
  // Walk through the cast path to create nested CXXBaseRegions.
  SVal Result = Derived;
  for (CastExpr::path_const_iterator I = Cast->path_begin(),
                                     E = Cast->path_end();
       I != E; ++I) {
    Result = evalDerivedToBase(Result, (*I)->getType());
  }
  return Result;
}

SVal StoreManager::evalDerivedToBase(SVal Derived, const CXXBasePath &Path) {
  // Walk through the path to create nested CXXBaseRegions.
  SVal Result = Derived;
  for (CXXBasePath::const_iterator I = Path.begin(), E = Path.end();
       I != E; ++I) {
    Result = evalDerivedToBase(Result, I->Base->getType());
  }
  return Result;
}

SVal StoreManager::evalDerivedToBase(SVal Derived, QualType BaseType) {
  loc::MemRegionVal *DerivedRegVal = dyn_cast<loc::MemRegionVal>(&Derived);
  if (!DerivedRegVal)
    return Derived;

  const CXXRecordDecl *BaseDecl = BaseType->getPointeeCXXRecordDecl();
  if (!BaseDecl)
    BaseDecl = BaseType->getAsCXXRecordDecl();
  assert(BaseDecl && "not a C++ object?");

  const MemRegion *BaseReg =
    MRMgr.getCXXBaseObjectRegion(BaseDecl, DerivedRegVal->getRegion());

  return loc::MemRegionVal(BaseReg);
}

SVal StoreManager::evalDynamicCast(SVal Base, QualType DerivedType,
                                   bool &Failed) {
  Failed = false;

  loc::MemRegionVal *BaseRegVal = dyn_cast<loc::MemRegionVal>(&Base);
  if (!BaseRegVal)
    return UnknownVal();
  const MemRegion *BaseRegion = BaseRegVal->stripCasts(/*StripBases=*/false);

  // Assume the derived class is a pointer or a reference to a CXX record.
  DerivedType = DerivedType->getPointeeType();
  assert(!DerivedType.isNull());
  const CXXRecordDecl *DerivedDecl = DerivedType->getAsCXXRecordDecl();
  if (!DerivedDecl && !DerivedType->isVoidType())
    return UnknownVal();

  // Drill down the CXXBaseObject chains, which represent upcasts (casts from
  // derived to base).
  const MemRegion *SR = BaseRegion;
  while (const TypedRegion *TSR = dyn_cast_or_null<TypedRegion>(SR)) {
    QualType BaseType = TSR->getLocationType()->getPointeeType();
    assert(!BaseType.isNull());
    const CXXRecordDecl *SRDecl = BaseType->getAsCXXRecordDecl();
    if (!SRDecl)
      return UnknownVal();

    // If found the derived class, the cast succeeds.
    if (SRDecl == DerivedDecl)
      return loc::MemRegionVal(TSR);

    if (!DerivedType->isVoidType()) {
      // Static upcasts are marked as DerivedToBase casts by Sema, so this will
      // only happen when multiple or virtual inheritance is involved.
      CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/true,
                         /*DetectVirtual=*/false);
      if (SRDecl->isDerivedFrom(DerivedDecl, Paths))
        return evalDerivedToBase(loc::MemRegionVal(TSR), Paths.front());
    }

    if (const CXXBaseObjectRegion *R = dyn_cast<CXXBaseObjectRegion>(TSR))
      // Drill down the chain to get the derived classes.
      SR = R->getSuperRegion();
    else {
      // We reached the bottom of the hierarchy.

      // If this is a cast to void*, return the region.
      if (DerivedType->isVoidType())
        return loc::MemRegionVal(TSR);

      // We did not find the derived class. We we must be casting the base to
      // derived, so the cast should fail.
      Failed = true;
      return UnknownVal();
    }
  }
  
  return UnknownVal();
}


/// CastRetrievedVal - Used by subclasses of StoreManager to implement
///  implicit casts that arise from loads from regions that are reinterpreted
///  as another region.
SVal StoreManager::CastRetrievedVal(SVal V, const TypedValueRegion *R,
                                    QualType castTy, bool performTestOnly) {
  
  if (castTy.isNull() || V.isUnknownOrUndef())
    return V;
  
  ASTContext &Ctx = svalBuilder.getContext();

  if (performTestOnly) {  
    // Automatically translate references to pointers.
    QualType T = R->getValueType();
    if (const ReferenceType *RT = T->getAs<ReferenceType>())
      T = Ctx.getPointerType(RT->getPointeeType());
    
    assert(svalBuilder.getContext().hasSameUnqualifiedType(castTy, T));
    return V;
  }
  
  return svalBuilder.dispatchCast(V, castTy);
}

SVal StoreManager::getLValueFieldOrIvar(const Decl *D, SVal Base) {
  if (Base.isUnknownOrUndef())
    return Base;

  Loc BaseL = cast<Loc>(Base);
  const MemRegion* BaseR = 0;

  switch (BaseL.getSubKind()) {
  case loc::MemRegionKind:
    BaseR = cast<loc::MemRegionVal>(BaseL).getRegion();
    break;

  case loc::GotoLabelKind:
    // These are anormal cases. Flag an undefined value.
    return UndefinedVal();

  case loc::ConcreteIntKind:
    // While these seem funny, this can happen through casts.
    // FIXME: What we should return is the field offset.  For example,
    //  add the field offset to the integer value.  That way funny things
    //  like this work properly:  &(((struct foo *) 0xa)->f)
    return Base;

  default:
    llvm_unreachable("Unhandled Base.");
  }

  // NOTE: We must have this check first because ObjCIvarDecl is a subclass
  // of FieldDecl.
  if (const ObjCIvarDecl *ID = dyn_cast<ObjCIvarDecl>(D))
    return loc::MemRegionVal(MRMgr.getObjCIvarRegion(ID, BaseR));

  return loc::MemRegionVal(MRMgr.getFieldRegion(cast<FieldDecl>(D), BaseR));
}

SVal StoreManager::getLValueIvar(const ObjCIvarDecl *decl, SVal base) {
  return getLValueFieldOrIvar(decl, base);
}

SVal StoreManager::getLValueElement(QualType elementType, NonLoc Offset, 
                                    SVal Base) {

  // If the base is an unknown or undefined value, just return it back.
  // FIXME: For absolute pointer addresses, we just return that value back as
  //  well, although in reality we should return the offset added to that
  //  value.
  if (Base.isUnknownOrUndef() || isa<loc::ConcreteInt>(Base))
    return Base;

  const MemRegion* BaseRegion = cast<loc::MemRegionVal>(Base).getRegion();

  // Pointer of any type can be cast and used as array base.
  const ElementRegion *ElemR = dyn_cast<ElementRegion>(BaseRegion);

  // Convert the offset to the appropriate size and signedness.
  Offset = cast<NonLoc>(svalBuilder.convertToArrayIndex(Offset));

  if (!ElemR) {
    //
    // If the base region is not an ElementRegion, create one.
    // This can happen in the following example:
    //
    //   char *p = __builtin_alloc(10);
    //   p[1] = 8;
    //
    //  Observe that 'p' binds to an AllocaRegion.
    //
    return loc::MemRegionVal(MRMgr.getElementRegion(elementType, Offset,
                                                    BaseRegion, Ctx));
  }

  SVal BaseIdx = ElemR->getIndex();

  if (!isa<nonloc::ConcreteInt>(BaseIdx))
    return UnknownVal();

  const llvm::APSInt& BaseIdxI = cast<nonloc::ConcreteInt>(BaseIdx).getValue();

  // Only allow non-integer offsets if the base region has no offset itself.
  // FIXME: This is a somewhat arbitrary restriction. We should be using
  // SValBuilder here to add the two offsets without checking their types.
  if (!isa<nonloc::ConcreteInt>(Offset)) {
    if (isa<ElementRegion>(BaseRegion->StripCasts()))
      return UnknownVal();

    return loc::MemRegionVal(MRMgr.getElementRegion(elementType, Offset,
                                                    ElemR->getSuperRegion(),
                                                    Ctx));
  }

  const llvm::APSInt& OffI = cast<nonloc::ConcreteInt>(Offset).getValue();
  assert(BaseIdxI.isSigned());

  // Compute the new index.
  nonloc::ConcreteInt NewIdx(svalBuilder.getBasicValueFactory().getValue(BaseIdxI +
                                                                    OffI));

  // Construct the new ElementRegion.
  const MemRegion *ArrayR = ElemR->getSuperRegion();
  return loc::MemRegionVal(MRMgr.getElementRegion(elementType, NewIdx, ArrayR,
                                                  Ctx));
}

StoreManager::BindingsHandler::~BindingsHandler() {}

bool StoreManager::FindUniqueBinding::HandleBinding(StoreManager& SMgr,
                                                    Store store,
                                                    const MemRegion* R,
                                                    SVal val) {
  SymbolRef SymV = val.getAsLocSymbol();
  if (!SymV || SymV != Sym)
    return true;

  if (Binding) {
    First = false;
    return false;
  }
  else
    Binding = R;

  return true;
}
