//===--- VTableBuilder.cpp - C++ vtable layout builder --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with generation of the layout of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/VTableBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <cstdio>

using namespace clang;

#define DUMP_OVERRIDERS 0

namespace {

/// BaseOffset - Represents an offset from a derived class to a direct or
/// indirect base class.
struct BaseOffset {
  /// DerivedClass - The derived class.
  const CXXRecordDecl *DerivedClass;
  
  /// VirtualBase - If the path from the derived class to the base class
  /// involves a virtual base class, this holds its declaration.
  const CXXRecordDecl *VirtualBase;

  /// NonVirtualOffset - The offset from the derived class to the base class.
  /// (Or the offset from the virtual base class to the base class, if the 
  /// path from the derived class to the base class involves a virtual base
  /// class.
  CharUnits NonVirtualOffset;
  
  BaseOffset() : DerivedClass(0), VirtualBase(0), 
    NonVirtualOffset(CharUnits::Zero()) { }
  BaseOffset(const CXXRecordDecl *DerivedClass,
             const CXXRecordDecl *VirtualBase, CharUnits NonVirtualOffset)
    : DerivedClass(DerivedClass), VirtualBase(VirtualBase), 
    NonVirtualOffset(NonVirtualOffset) { }

  bool isEmpty() const { return NonVirtualOffset.isZero() && !VirtualBase; }
};

/// FinalOverriders - Contains the final overrider member functions for all
/// member functions in the base subobjects of a class.
class FinalOverriders {
public:
  /// OverriderInfo - Information about a final overrider.
  struct OverriderInfo {
    /// Method - The method decl of the overrider.
    const CXXMethodDecl *Method;

    /// Offset - the base offset of the overrider in the layout class.
    CharUnits Offset;
    
    OverriderInfo() : Method(0), Offset(CharUnits::Zero()) { }
  };

private:
  /// MostDerivedClass - The most derived class for which the final overriders
  /// are stored.
  const CXXRecordDecl *MostDerivedClass;
  
  /// MostDerivedClassOffset - If we're building final overriders for a 
  /// construction vtable, this holds the offset from the layout class to the
  /// most derived class.
  const CharUnits MostDerivedClassOffset;

  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if the final overriders are for a
  /// construction vtable.  
  const CXXRecordDecl *LayoutClass;  

  ASTContext &Context;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  /// MethodBaseOffsetPairTy - Uniquely identifies a member function
  /// in a base subobject.
  typedef std::pair<const CXXMethodDecl *, CharUnits> MethodBaseOffsetPairTy;

  typedef llvm::DenseMap<MethodBaseOffsetPairTy,
                         OverriderInfo> OverridersMapTy;
  
  /// OverridersMap - The final overriders for all virtual member functions of 
  /// all the base subobjects of the most derived class.
  OverridersMapTy OverridersMap;
  
  /// SubobjectsToOffsetsMapTy - A mapping from a base subobject (represented
  /// as a record decl and a subobject number) and its offsets in the most
  /// derived class as well as the layout class.
  typedef llvm::DenseMap<std::pair<const CXXRecordDecl *, unsigned>, 
                         CharUnits> SubobjectOffsetMapTy;

  typedef llvm::DenseMap<const CXXRecordDecl *, unsigned> SubobjectCountMapTy;
  
  /// ComputeBaseOffsets - Compute the offsets for all base subobjects of the
  /// given base.
  void ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                          CharUnits OffsetInLayoutClass,
                          SubobjectOffsetMapTy &SubobjectOffsets,
                          SubobjectOffsetMapTy &SubobjectLayoutClassOffsets,
                          SubobjectCountMapTy &SubobjectCounts);

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;
  
  /// dump - dump the final overriders for a base subobject, and all its direct
  /// and indirect base subobjects.
  void dump(raw_ostream &Out, BaseSubobject Base,
            VisitedVirtualBasesSetTy& VisitedVirtualBases);
  
public:
  FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                  CharUnits MostDerivedClassOffset,
                  const CXXRecordDecl *LayoutClass);

  /// getOverrider - Get the final overrider for the given method declaration in
  /// the subobject with the given base offset. 
  OverriderInfo getOverrider(const CXXMethodDecl *MD, 
                             CharUnits BaseOffset) const {
    assert(OverridersMap.count(std::make_pair(MD, BaseOffset)) && 
           "Did not find overrider!");
    
    return OverridersMap.lookup(std::make_pair(MD, BaseOffset));
  }
  
  /// dump - dump the final overriders.
  void dump() {
    VisitedVirtualBasesSetTy VisitedVirtualBases;
    dump(llvm::errs(), BaseSubobject(MostDerivedClass, CharUnits::Zero()), 
         VisitedVirtualBases);
  }
  
};

#define DUMP_OVERRIDERS 0

FinalOverriders::FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                                 CharUnits MostDerivedClassOffset,
                                 const CXXRecordDecl *LayoutClass)
  : MostDerivedClass(MostDerivedClass), 
  MostDerivedClassOffset(MostDerivedClassOffset), LayoutClass(LayoutClass),
  Context(MostDerivedClass->getASTContext()),
  MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)) {

  // Compute base offsets.
  SubobjectOffsetMapTy SubobjectOffsets;
  SubobjectOffsetMapTy SubobjectLayoutClassOffsets;
  SubobjectCountMapTy SubobjectCounts;
  ComputeBaseOffsets(BaseSubobject(MostDerivedClass, CharUnits::Zero()), 
                     /*IsVirtual=*/false,
                     MostDerivedClassOffset, 
                     SubobjectOffsets, SubobjectLayoutClassOffsets, 
                     SubobjectCounts);

  // Get the final overriders.
  CXXFinalOverriderMap FinalOverriders;
  MostDerivedClass->getFinalOverriders(FinalOverriders);

  for (CXXFinalOverriderMap::const_iterator I = FinalOverriders.begin(),
       E = FinalOverriders.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const OverridingMethods& Methods = I->second;

    for (OverridingMethods::const_iterator I = Methods.begin(),
         E = Methods.end(); I != E; ++I) {
      unsigned SubobjectNumber = I->first;
      assert(SubobjectOffsets.count(std::make_pair(MD->getParent(), 
                                                   SubobjectNumber)) &&
             "Did not find subobject offset!");
      
      CharUnits BaseOffset = SubobjectOffsets[std::make_pair(MD->getParent(),
                                                            SubobjectNumber)];

      assert(I->second.size() == 1 && "Final overrider is not unique!");
      const UniqueVirtualMethod &Method = I->second.front();

      const CXXRecordDecl *OverriderRD = Method.Method->getParent();
      assert(SubobjectLayoutClassOffsets.count(
             std::make_pair(OverriderRD, Method.Subobject))
             && "Did not find subobject offset!");
      CharUnits OverriderOffset =
        SubobjectLayoutClassOffsets[std::make_pair(OverriderRD, 
                                                   Method.Subobject)];

      OverriderInfo& Overrider = OverridersMap[std::make_pair(MD, BaseOffset)];
      assert(!Overrider.Method && "Overrider should not exist yet!");
      
      Overrider.Offset = OverriderOffset;
      Overrider.Method = Method.Method;
    }
  }

#if DUMP_OVERRIDERS
  // And dump them (for now).
  dump();
#endif
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *DerivedRD,
                                    const CXXBasePath &Path) {
  CharUnits NonVirtualOffset = CharUnits::Zero();

  unsigned NonVirtualStart = 0;
  const CXXRecordDecl *VirtualBase = 0;
  
  // First, look for the virtual base class.
  for (unsigned I = 0, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement &Element = Path[I];
    
    if (Element.Base->isVirtual()) {
      // FIXME: Can we break when we find the first virtual base?
      // (If we can't, can't we just iterate over the path in reverse order?)
      NonVirtualStart = I + 1;
      QualType VBaseType = Element.Base->getType();
      VirtualBase = 
        cast<CXXRecordDecl>(VBaseType->getAs<RecordType>()->getDecl());
    }
  }
  
  // Now compute the non-virtual offset.
  for (unsigned I = NonVirtualStart, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement &Element = Path[I];
    
    // Check the base class offset.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);

    const RecordType *BaseType = Element.Base->getType()->getAs<RecordType>();
    const CXXRecordDecl *Base = cast<CXXRecordDecl>(BaseType->getDecl());

    NonVirtualOffset += Layout.getBaseClassOffset(Base);
  }
  
  // FIXME: This should probably use CharUnits or something. Maybe we should
  // even change the base offsets in ASTRecordLayout to be specified in 
  // CharUnits.
  return BaseOffset(DerivedRD, VirtualBase, NonVirtualOffset);
  
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *BaseRD,
                                    const CXXRecordDecl *DerivedRD) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  
  if (!const_cast<CXXRecordDecl *>(DerivedRD)->
      isDerivedFrom(const_cast<CXXRecordDecl *>(BaseRD), Paths)) {
    llvm_unreachable("Class must be derived from the passed in base class!");
  }

  return ComputeBaseOffset(Context, DerivedRD, Paths.front());
}

static BaseOffset
ComputeReturnAdjustmentBaseOffset(ASTContext &Context, 
                                  const CXXMethodDecl *DerivedMD,
                                  const CXXMethodDecl *BaseMD) {
  const FunctionType *BaseFT = BaseMD->getType()->getAs<FunctionType>();
  const FunctionType *DerivedFT = DerivedMD->getType()->getAs<FunctionType>();
  
  // Canonicalize the return types.
  CanQualType CanDerivedReturnType = 
    Context.getCanonicalType(DerivedFT->getResultType());
  CanQualType CanBaseReturnType = 
    Context.getCanonicalType(BaseFT->getResultType());
  
  assert(CanDerivedReturnType->getTypeClass() == 
         CanBaseReturnType->getTypeClass() && 
         "Types must have same type class!");
  
  if (CanDerivedReturnType == CanBaseReturnType) {
    // No adjustment needed.
    return BaseOffset();
  }
  
  if (isa<ReferenceType>(CanDerivedReturnType)) {
    CanDerivedReturnType = 
      CanDerivedReturnType->getAs<ReferenceType>()->getPointeeType();
    CanBaseReturnType = 
      CanBaseReturnType->getAs<ReferenceType>()->getPointeeType();
  } else if (isa<PointerType>(CanDerivedReturnType)) {
    CanDerivedReturnType = 
      CanDerivedReturnType->getAs<PointerType>()->getPointeeType();
    CanBaseReturnType = 
      CanBaseReturnType->getAs<PointerType>()->getPointeeType();
  } else {
    llvm_unreachable("Unexpected return type!");
  }
  
  // We need to compare unqualified types here; consider
  //   const T *Base::foo();
  //   T *Derived::foo();
  if (CanDerivedReturnType.getUnqualifiedType() == 
      CanBaseReturnType.getUnqualifiedType()) {
    // No adjustment needed.
    return BaseOffset();
  }
  
  const CXXRecordDecl *DerivedRD = 
    cast<CXXRecordDecl>(cast<RecordType>(CanDerivedReturnType)->getDecl());
  
  const CXXRecordDecl *BaseRD = 
    cast<CXXRecordDecl>(cast<RecordType>(CanBaseReturnType)->getDecl());

  return ComputeBaseOffset(Context, BaseRD, DerivedRD);
}

void 
FinalOverriders::ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                              CharUnits OffsetInLayoutClass,
                              SubobjectOffsetMapTy &SubobjectOffsets,
                              SubobjectOffsetMapTy &SubobjectLayoutClassOffsets,
                              SubobjectCountMapTy &SubobjectCounts) {
  const CXXRecordDecl *RD = Base.getBase();
  
  unsigned SubobjectNumber = 0;
  if (!IsVirtual)
    SubobjectNumber = ++SubobjectCounts[RD];

  // Set up the subobject to offset mapping.
  assert(!SubobjectOffsets.count(std::make_pair(RD, SubobjectNumber))
         && "Subobject offset already exists!");
  assert(!SubobjectLayoutClassOffsets.count(std::make_pair(RD, SubobjectNumber)) 
         && "Subobject offset already exists!");

  SubobjectOffsets[std::make_pair(RD, SubobjectNumber)] = Base.getBaseOffset();
  SubobjectLayoutClassOffsets[std::make_pair(RD, SubobjectNumber)] =
    OffsetInLayoutClass;
  
  // Traverse our bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset;
    CharUnits BaseOffsetInLayoutClass;
    if (I->isVirtual()) {
      // Check if we've visited this virtual base before.
      if (SubobjectOffsets.count(std::make_pair(BaseDecl, 0)))
        continue;

      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
      CharUnits Offset = Layout.getBaseClassOffset(BaseDecl);
    
      BaseOffset = Base.getBaseOffset() + Offset;
      BaseOffsetInLayoutClass = OffsetInLayoutClass + Offset;
    }

    ComputeBaseOffsets(BaseSubobject(BaseDecl, BaseOffset), 
                       I->isVirtual(), BaseOffsetInLayoutClass, 
                       SubobjectOffsets, SubobjectLayoutClassOffsets, 
                       SubobjectCounts);
  }
}

void FinalOverriders::dump(raw_ostream &Out, BaseSubobject Base,
                           VisitedVirtualBasesSetTy &VisitedVirtualBases) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    // Ignore bases that don't have any virtual member functions.
    if (!BaseDecl->isPolymorphic())
      continue;

    CharUnits BaseOffset;
    if (I->isVirtual()) {
      if (!VisitedVirtualBases.insert(BaseDecl)) {
        // We've visited this base before.
        continue;
      }
      
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffset = Layout.getBaseClassOffset(BaseDecl) + Base.getBaseOffset();
    }

    dump(Out, BaseSubobject(BaseDecl, BaseOffset), VisitedVirtualBases);
  }

  Out << "Final overriders for (" << RD->getQualifiedNameAsString() << ", ";
  Out << Base.getBaseOffset().getQuantity() << ")\n";

  // Now dump the overriders for this base subobject.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;
  
    OverriderInfo Overrider = getOverrider(MD, Base.getBaseOffset());

    Out << "  " << MD->getQualifiedNameAsString() << " - (";
    Out << Overrider.Method->getQualifiedNameAsString();
    Out << ", " << ", " << Overrider.Offset.getQuantity() << ')';

    BaseOffset Offset;
    if (!Overrider.Method->isPure())
      Offset = ComputeReturnAdjustmentBaseOffset(Context, Overrider.Method, MD);

    if (!Offset.isEmpty()) {
      Out << " [ret-adj: ";
      if (Offset.VirtualBase)
        Out << Offset.VirtualBase->getQualifiedNameAsString() << " vbase, ";
             
      Out << Offset.NonVirtualOffset.getQuantity() << " nv]";
    }
    
    Out << "\n";
  }  
}

/// VCallOffsetMap - Keeps track of vcall offsets when building a vtable.
struct VCallOffsetMap {
  
  typedef std::pair<const CXXMethodDecl *, CharUnits> MethodAndOffsetPairTy;
  
  /// Offsets - Keeps track of methods and their offsets.
  // FIXME: This should be a real map and not a vector.
  SmallVector<MethodAndOffsetPairTy, 16> Offsets;

  /// MethodsCanShareVCallOffset - Returns whether two virtual member functions
  /// can share the same vcall offset.
  static bool MethodsCanShareVCallOffset(const CXXMethodDecl *LHS,
                                         const CXXMethodDecl *RHS);

public:
  /// AddVCallOffset - Adds a vcall offset to the map. Returns true if the
  /// add was successful, or false if there was already a member function with
  /// the same signature in the map.
  bool AddVCallOffset(const CXXMethodDecl *MD, CharUnits OffsetOffset);
  
  /// getVCallOffsetOffset - Returns the vcall offset offset (relative to the
  /// vtable address point) for the given virtual member function.
  CharUnits getVCallOffsetOffset(const CXXMethodDecl *MD);
  
  // empty - Return whether the offset map is empty or not.
  bool empty() const { return Offsets.empty(); }
};

static bool HasSameVirtualSignature(const CXXMethodDecl *LHS,
                                    const CXXMethodDecl *RHS) {
  const FunctionProtoType *LT =
    cast<FunctionProtoType>(LHS->getType().getCanonicalType());
  const FunctionProtoType *RT =
    cast<FunctionProtoType>(RHS->getType().getCanonicalType());

  // Fast-path matches in the canonical types.
  if (LT == RT) return true;

  // Force the signatures to match.  We can't rely on the overrides
  // list here because there isn't necessarily an inheritance
  // relationship between the two methods.
  if (LT->getTypeQuals() != RT->getTypeQuals() ||
      LT->getNumArgs() != RT->getNumArgs())
    return false;
  for (unsigned I = 0, E = LT->getNumArgs(); I != E; ++I)
    if (LT->getArgType(I) != RT->getArgType(I))
      return false;
  return true;
}

bool VCallOffsetMap::MethodsCanShareVCallOffset(const CXXMethodDecl *LHS,
                                                const CXXMethodDecl *RHS) {
  assert(LHS->isVirtual() && "LHS must be virtual!");
  assert(RHS->isVirtual() && "LHS must be virtual!");
  
  // A destructor can share a vcall offset with another destructor.
  if (isa<CXXDestructorDecl>(LHS))
    return isa<CXXDestructorDecl>(RHS);

  // FIXME: We need to check more things here.
  
  // The methods must have the same name.
  DeclarationName LHSName = LHS->getDeclName();
  DeclarationName RHSName = RHS->getDeclName();
  if (LHSName != RHSName)
    return false;

  // And the same signatures.
  return HasSameVirtualSignature(LHS, RHS);
}

bool VCallOffsetMap::AddVCallOffset(const CXXMethodDecl *MD, 
                                    CharUnits OffsetOffset) {
  // Check if we can reuse an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return false;
  }
  
  // Add the offset.
  Offsets.push_back(MethodAndOffsetPairTy(MD, OffsetOffset));
  return true;
}

CharUnits VCallOffsetMap::getVCallOffsetOffset(const CXXMethodDecl *MD) {
  // Look for an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return Offsets[I].second;
  }
  
  llvm_unreachable("Should always find a vcall offset offset!");
}

/// VCallAndVBaseOffsetBuilder - Class for building vcall and vbase offsets.
class VCallAndVBaseOffsetBuilder {
public:
  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> 
    VBaseOffsetOffsetsMapTy;

private:
  /// MostDerivedClass - The most derived class for which we're building vcall
  /// and vbase offsets.
  const CXXRecordDecl *MostDerivedClass;
  
  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if we're building a construction
  /// vtable.
  const CXXRecordDecl *LayoutClass;
  
  /// Context - The ASTContext which we will use for layout information.
  ASTContext &Context;

  /// Components - vcall and vbase offset components
  typedef SmallVector<VTableComponent, 64> VTableComponentVectorTy;
  VTableComponentVectorTy Components;
  
  /// VisitedVirtualBases - Visited virtual bases.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;
  
  /// VCallOffsets - Keeps track of vcall offsets.
  VCallOffsetMap VCallOffsets;


  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets,
  /// relative to the address point.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// FinalOverriders - The final overriders of the most derived class.
  /// (Can be null when we're not building a vtable of the most derived class).
  const FinalOverriders *Overriders;

  /// AddVCallAndVBaseOffsets - Add vcall offsets and vbase offsets for the
  /// given base subobject.
  void AddVCallAndVBaseOffsets(BaseSubobject Base, bool BaseIsVirtual,
                               CharUnits RealBaseOffset);
  
  /// AddVCallOffsets - Add vcall offsets for the given base subobject.
  void AddVCallOffsets(BaseSubobject Base, CharUnits VBaseOffset);
  
  /// AddVBaseOffsets - Add vbase offsets for the given class.
  void AddVBaseOffsets(const CXXRecordDecl *Base, 
                       CharUnits OffsetInLayoutClass);
  
  /// getCurrentOffsetOffset - Get the current vcall or vbase offset offset in
  /// chars, relative to the vtable address point.
  CharUnits getCurrentOffsetOffset() const;
  
public:
  VCallAndVBaseOffsetBuilder(const CXXRecordDecl *MostDerivedClass,
                             const CXXRecordDecl *LayoutClass,
                             const FinalOverriders *Overriders,
                             BaseSubobject Base, bool BaseIsVirtual,
                             CharUnits OffsetInLayoutClass)
    : MostDerivedClass(MostDerivedClass), LayoutClass(LayoutClass), 
    Context(MostDerivedClass->getASTContext()), Overriders(Overriders) {
      
    // Add vcall and vbase offsets.
    AddVCallAndVBaseOffsets(Base, BaseIsVirtual, OffsetInLayoutClass);
  }
  
  /// Methods for iterating over the components.
  typedef VTableComponentVectorTy::const_reverse_iterator const_iterator;
  const_iterator components_begin() const { return Components.rbegin(); }
  const_iterator components_end() const { return Components.rend(); }
  
  const VCallOffsetMap &getVCallOffsets() const { return VCallOffsets; }
  const VBaseOffsetOffsetsMapTy &getVBaseOffsetOffsets() const {
    return VBaseOffsetOffsets;
  }
};
  
void 
VCallAndVBaseOffsetBuilder::AddVCallAndVBaseOffsets(BaseSubobject Base,
                                                    bool BaseIsVirtual,
                                                    CharUnits RealBaseOffset) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Base.getBase());
  
  // Itanium C++ ABI 2.5.2:
  //   ..in classes sharing a virtual table with a primary base class, the vcall
  //   and vbase offsets added by the derived class all come before the vcall
  //   and vbase offsets required by the base class, so that the latter may be
  //   laid out as required by the base class without regard to additions from
  //   the derived class(es).

  // (Since we're emitting the vcall and vbase offsets in reverse order, we'll
  // emit them for the primary base first).
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    bool PrimaryBaseIsVirtual = Layout.isPrimaryBaseVirtual();

    CharUnits PrimaryBaseOffset;
    
    // Get the base offset of the primary base.
    if (PrimaryBaseIsVirtual) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
    }

    AddVCallAndVBaseOffsets(
      BaseSubobject(PrimaryBase,PrimaryBaseOffset),
      PrimaryBaseIsVirtual, RealBaseOffset);
  }

  AddVBaseOffsets(Base.getBase(), RealBaseOffset);

  // We only want to add vcall offsets for virtual bases.
  if (BaseIsVirtual)
    AddVCallOffsets(Base, RealBaseOffset);
}

CharUnits VCallAndVBaseOffsetBuilder::getCurrentOffsetOffset() const {
  // OffsetIndex is the index of this vcall or vbase offset, relative to the 
  // vtable address point. (We subtract 3 to account for the information just
  // above the address point, the RTTI info, the offset to top, and the
  // vcall offset itself).
  int64_t OffsetIndex = -(int64_t)(3 + Components.size());
    
  CharUnits PointerWidth = 
    Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
  CharUnits OffsetOffset = PointerWidth * OffsetIndex;
  return OffsetOffset;
}

void VCallAndVBaseOffsetBuilder::AddVCallOffsets(BaseSubobject Base, 
                                                 CharUnits VBaseOffset) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // Handle the primary base first.
  // We only want to add vcall offsets if the base is non-virtual; a virtual
  // primary base will have its vcall and vbase offsets emitted already.
  if (PrimaryBase && !Layout.isPrimaryBaseVirtual()) {
    // Get the base offset of the primary base.
    assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
           "Primary base should have a zero offset!");

    AddVCallOffsets(BaseSubobject(PrimaryBase, Base.getBaseOffset()),
                    VBaseOffset);
  }
  
  // Add the vcall offsets.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;

    CharUnits OffsetOffset = getCurrentOffsetOffset();
    
    // Don't add a vcall offset if we already have one for this member function
    // signature.
    if (!VCallOffsets.AddVCallOffset(MD, OffsetOffset))
      continue;

    CharUnits Offset = CharUnits::Zero();

    if (Overriders) {
      // Get the final overrider.
      FinalOverriders::OverriderInfo Overrider = 
        Overriders->getOverrider(MD, Base.getBaseOffset());
      
      /// The vcall offset is the offset from the virtual base to the object 
      /// where the function was overridden.
      Offset = Overrider.Offset - VBaseOffset;
    }
    
    Components.push_back(
      VTableComponent::MakeVCallOffset(Offset));
  }

  // And iterate over all non-virtual bases (ignoring the primary base).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
  
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    if (BaseDecl == PrimaryBase)
      continue;

    // Get the base offset of this base.
    CharUnits BaseOffset = Base.getBaseOffset() + 
      Layout.getBaseClassOffset(BaseDecl);
    
    AddVCallOffsets(BaseSubobject(BaseDecl, BaseOffset), 
                    VBaseOffset);
  }
}

void 
VCallAndVBaseOffsetBuilder::AddVBaseOffsets(const CXXRecordDecl *RD,
                                            CharUnits OffsetInLayoutClass) {
  const ASTRecordLayout &LayoutClassLayout = 
    Context.getASTRecordLayout(LayoutClass);

  // Add vbase offsets.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this is a virtual base that we haven't visited before.
    if (I->isVirtual() && VisitedVirtualBases.insert(BaseDecl)) {
      CharUnits Offset = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl) - OffsetInLayoutClass;

      // Add the vbase offset offset.
      assert(!VBaseOffsetOffsets.count(BaseDecl) &&
             "vbase offset offset already exists!");

      CharUnits VBaseOffsetOffset = getCurrentOffsetOffset();
      VBaseOffsetOffsets.insert(
          std::make_pair(BaseDecl, VBaseOffsetOffset));

      Components.push_back(
          VTableComponent::MakeVBaseOffset(Offset));
    }

    // Check the base class looking for more vbase offsets.
    AddVBaseOffsets(BaseDecl, OffsetInLayoutClass);
  }
}

/// VTableBuilder - Class for building vtable layout information.
class VTableBuilder {
public:
  /// PrimaryBasesSetVectorTy - A set vector of direct and indirect 
  /// primary bases.
  typedef llvm::SmallSetVector<const CXXRecordDecl *, 8> 
    PrimaryBasesSetVectorTy;
  
  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> 
    VBaseOffsetOffsetsMapTy;
  
  typedef llvm::DenseMap<BaseSubobject, uint64_t> 
    AddressPointsMapTy;

private:
  /// VTables - Global vtable information.
  VTableContext &VTables;
  
  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  /// MostDerivedClassOffset - If we're building a construction vtable, this
  /// holds the offset from the layout class to the most derived class.
  const CharUnits MostDerivedClassOffset;
  
  /// MostDerivedClassIsVirtual - Whether the most derived class is a virtual 
  /// base. (This only makes sense when building a construction vtable).
  bool MostDerivedClassIsVirtual;
  
  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if we're building a construction
  /// vtable.
  const CXXRecordDecl *LayoutClass;
  
  /// Context - The ASTContext which we will use for layout information.
  ASTContext &Context;
  
  /// FinalOverriders - The final overriders of the most derived class.
  const FinalOverriders Overriders;

  /// VCallOffsetsForVBases - Keeps track of vcall offsets for the virtual
  /// bases in this vtable.
  llvm::DenseMap<const CXXRecordDecl *, VCallOffsetMap> VCallOffsetsForVBases;

  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets for
  /// the most derived class.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// Components - The components of the vtable being built.
  SmallVector<VTableComponent, 64> Components;

  /// AddressPoints - Address points for the vtable being built.
  AddressPointsMapTy AddressPoints;

  /// MethodInfo - Contains information about a method in a vtable.
  /// (Used for computing 'this' pointer adjustment thunks.
  struct MethodInfo {
    /// BaseOffset - The base offset of this method.
    const CharUnits BaseOffset;
    
    /// BaseOffsetInLayoutClass - The base offset in the layout class of this
    /// method.
    const CharUnits BaseOffsetInLayoutClass;
    
    /// VTableIndex - The index in the vtable that this method has.
    /// (For destructors, this is the index of the complete destructor).
    const uint64_t VTableIndex;
    
    MethodInfo(CharUnits BaseOffset, CharUnits BaseOffsetInLayoutClass, 
               uint64_t VTableIndex)
      : BaseOffset(BaseOffset), 
      BaseOffsetInLayoutClass(BaseOffsetInLayoutClass),
      VTableIndex(VTableIndex) { }
    
    MethodInfo() 
      : BaseOffset(CharUnits::Zero()), 
      BaseOffsetInLayoutClass(CharUnits::Zero()), 
      VTableIndex(0) { }
  };
  
  typedef llvm::DenseMap<const CXXMethodDecl *, MethodInfo> MethodInfoMapTy;
  
  /// MethodInfoMap - The information for all methods in the vtable we're
  /// currently building.
  MethodInfoMapTy MethodInfoMap;
  
  typedef llvm::DenseMap<uint64_t, ThunkInfo> VTableThunksMapTy;
  
  /// VTableThunks - The thunks by vtable index in the vtable currently being 
  /// built.
  VTableThunksMapTy VTableThunks;

  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;
  
  /// Thunks - A map that contains all the thunks needed for all methods in the
  /// most derived class for which the vtable is currently being built.
  ThunksMapTy Thunks;
  
  /// AddThunk - Add a thunk for the given method.
  void AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk);
  
  /// ComputeThisAdjustments - Compute the 'this' pointer adjustments for the
  /// part of the vtable we're currently building.
  void ComputeThisAdjustments();
  
  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;

  /// PrimaryVirtualBases - All known virtual bases who are a primary base of
  /// some other base.
  VisitedVirtualBasesSetTy PrimaryVirtualBases;

  /// ComputeReturnAdjustment - Compute the return adjustment given a return
  /// adjustment base offset.
  ReturnAdjustment ComputeReturnAdjustment(BaseOffset Offset);
  
  /// ComputeThisAdjustmentBaseOffset - Compute the base offset for adjusting
  /// the 'this' pointer from the base subobject to the derived subobject.
  BaseOffset ComputeThisAdjustmentBaseOffset(BaseSubobject Base,
                                             BaseSubobject Derived) const;

  /// ComputeThisAdjustment - Compute the 'this' pointer adjustment for the
  /// given virtual member function, its offset in the layout class and its
  /// final overrider.
  ThisAdjustment 
  ComputeThisAdjustment(const CXXMethodDecl *MD, 
                        CharUnits BaseOffsetInLayoutClass,
                        FinalOverriders::OverriderInfo Overrider);

  /// AddMethod - Add a single virtual member function to the vtable
  /// components vector.
  void AddMethod(const CXXMethodDecl *MD, ReturnAdjustment ReturnAdjustment);

  /// IsOverriderUsed - Returns whether the overrider will ever be used in this
  /// part of the vtable. 
  ///
  /// Itanium C++ ABI 2.5.2:
  ///
  ///   struct A { virtual void f(); };
  ///   struct B : virtual public A { int i; };
  ///   struct C : virtual public A { int j; };
  ///   struct D : public B, public C {};
  ///
  ///   When B and C are declared, A is a primary base in each case, so although
  ///   vcall offsets are allocated in the A-in-B and A-in-C vtables, no this
  ///   adjustment is required and no thunk is generated. However, inside D
  ///   objects, A is no longer a primary base of C, so if we allowed calls to
  ///   C::f() to use the copy of A's vtable in the C subobject, we would need
  ///   to adjust this from C* to B::A*, which would require a third-party 
  ///   thunk. Since we require that a call to C::f() first convert to A*, 
  ///   C-in-D's copy of A's vtable is never referenced, so this is not 
  ///   necessary.
  bool IsOverriderUsed(const CXXMethodDecl *Overrider,
                       CharUnits BaseOffsetInLayoutClass,
                       const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                       CharUnits FirstBaseOffsetInLayoutClass) const;

  
  /// AddMethods - Add the methods of this base subobject and all its
  /// primary bases to the vtable components vector.
  void AddMethods(BaseSubobject Base, CharUnits BaseOffsetInLayoutClass,
                  const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                  CharUnits FirstBaseOffsetInLayoutClass,
                  PrimaryBasesSetVectorTy &PrimaryBases);

  // LayoutVTable - Layout the vtable for the given base class, including its
  // secondary vtables and any vtables for virtual bases.
  void LayoutVTable();

  /// LayoutPrimaryAndSecondaryVTables - Layout the primary vtable for the
  /// given base subobject, as well as all its secondary vtables.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  ///
  /// \param BaseIsVirtualInLayoutClass - Whether the base subobject is virtual
  /// in the layout class. 
  void LayoutPrimaryAndSecondaryVTables(BaseSubobject Base,
                                        bool BaseIsMorallyVirtual,
                                        bool BaseIsVirtualInLayoutClass,
                                        CharUnits OffsetInLayoutClass);
  
  /// LayoutSecondaryVTables - Layout the secondary vtables for the given base
  /// subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  void LayoutSecondaryVTables(BaseSubobject Base, bool BaseIsMorallyVirtual,
                              CharUnits OffsetInLayoutClass);

  /// DeterminePrimaryVirtualBases - Determine the primary virtual bases in this
  /// class hierarchy.
  void DeterminePrimaryVirtualBases(const CXXRecordDecl *RD, 
                                    CharUnits OffsetInLayoutClass,
                                    VisitedVirtualBasesSetTy &VBases);

  /// LayoutVTablesForVirtualBases - Layout vtables for all virtual bases of the
  /// given base (excluding any primary bases).
  void LayoutVTablesForVirtualBases(const CXXRecordDecl *RD, 
                                    VisitedVirtualBasesSetTy &VBases);

  /// isBuildingConstructionVTable - Return whether this vtable builder is
  /// building a construction vtable.
  bool isBuildingConstructorVTable() const { 
    return MostDerivedClass != LayoutClass;
  }

public:
  VTableBuilder(VTableContext &VTables, const CXXRecordDecl *MostDerivedClass,
                CharUnits MostDerivedClassOffset, 
                bool MostDerivedClassIsVirtual, const 
                CXXRecordDecl *LayoutClass)
    : VTables(VTables), MostDerivedClass(MostDerivedClass),
    MostDerivedClassOffset(MostDerivedClassOffset), 
    MostDerivedClassIsVirtual(MostDerivedClassIsVirtual), 
    LayoutClass(LayoutClass), Context(MostDerivedClass->getASTContext()), 
    Overriders(MostDerivedClass, MostDerivedClassOffset, LayoutClass) {

    LayoutVTable();

    if (Context.getLangOpts().DumpVTableLayouts)
      dumpLayout(llvm::errs());
  }

  uint64_t getNumThunks() const {
    return Thunks.size();
  }

  ThunksMapTy::const_iterator thunks_begin() const {
    return Thunks.begin();
  }

  ThunksMapTy::const_iterator thunks_end() const {
    return Thunks.end();
  }

  const VBaseOffsetOffsetsMapTy &getVBaseOffsetOffsets() const {
    return VBaseOffsetOffsets;
  }

  const AddressPointsMapTy &getAddressPoints() const {
    return AddressPoints;
  }

  /// getNumVTableComponents - Return the number of components in the vtable
  /// currently built.
  uint64_t getNumVTableComponents() const {
    return Components.size();
  }

  const VTableComponent *vtable_component_begin() const {
    return Components.begin();
  }
  
  const VTableComponent *vtable_component_end() const {
    return Components.end();
  }
  
  AddressPointsMapTy::const_iterator address_points_begin() const {
    return AddressPoints.begin();
  }

  AddressPointsMapTy::const_iterator address_points_end() const {
    return AddressPoints.end();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_begin() const {
    return VTableThunks.begin();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_end() const {
    return VTableThunks.end();
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(raw_ostream&);
};

void VTableBuilder::AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk) {
  assert(!isBuildingConstructorVTable() && 
         "Can't add thunks for construction vtable");

  SmallVector<ThunkInfo, 1> &ThunksVector = Thunks[MD];
  
  // Check if we have this thunk already.
  if (std::find(ThunksVector.begin(), ThunksVector.end(), Thunk) != 
      ThunksVector.end())
    return;
  
  ThunksVector.push_back(Thunk);
}

typedef llvm::SmallPtrSet<const CXXMethodDecl *, 8> OverriddenMethodsSetTy;

/// ComputeAllOverriddenMethods - Given a method decl, will return a set of all
/// the overridden methods that the function decl overrides.
static void 
ComputeAllOverriddenMethods(const CXXMethodDecl *MD,
                            OverriddenMethodsSetTy& OverriddenMethods) {
  assert(MD->isVirtual() && "Method is not virtual!");

  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    
    OverriddenMethods.insert(OverriddenMD);
    
    ComputeAllOverriddenMethods(OverriddenMD, OverriddenMethods);
  }
}

void VTableBuilder::ComputeThisAdjustments() {
  // Now go through the method info map and see if any of the methods need
  // 'this' pointer adjustments.
  for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
       E = MethodInfoMap.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const MethodInfo &MethodInfo = I->second;

    // Ignore adjustments for unused function pointers.
    uint64_t VTableIndex = MethodInfo.VTableIndex;
    if (Components[VTableIndex].getKind() == 
        VTableComponent::CK_UnusedFunctionPointer)
      continue;
    
    // Get the final overrider for this method.
    FinalOverriders::OverriderInfo Overrider =
      Overriders.getOverrider(MD, MethodInfo.BaseOffset);
    
    // Check if we need an adjustment at all.
    if (MethodInfo.BaseOffsetInLayoutClass == Overrider.Offset) {
      // When a return thunk is needed by a derived class that overrides a
      // virtual base, gcc uses a virtual 'this' adjustment as well. 
      // While the thunk itself might be needed by vtables in subclasses or
      // in construction vtables, there doesn't seem to be a reason for using
      // the thunk in this vtable. Still, we do so to match gcc.
      if (VTableThunks.lookup(VTableIndex).Return.isEmpty())
        continue;
    }

    ThisAdjustment ThisAdjustment =
      ComputeThisAdjustment(MD, MethodInfo.BaseOffsetInLayoutClass, Overrider);

    if (ThisAdjustment.isEmpty())
      continue;

    // Add it.
    VTableThunks[VTableIndex].This = ThisAdjustment;

    if (isa<CXXDestructorDecl>(MD)) {
      // Add an adjustment for the deleting destructor as well.
      VTableThunks[VTableIndex + 1].This = ThisAdjustment;
    }
  }

  /// Clear the method info map.
  MethodInfoMap.clear();
  
  if (isBuildingConstructorVTable()) {
    // We don't need to store thunk information for construction vtables.
    return;
  }

  for (VTableThunksMapTy::const_iterator I = VTableThunks.begin(),
       E = VTableThunks.end(); I != E; ++I) {
    const VTableComponent &Component = Components[I->first];
    const ThunkInfo &Thunk = I->second;
    const CXXMethodDecl *MD;
    
    switch (Component.getKind()) {
    default:
      llvm_unreachable("Unexpected vtable component kind!");
    case VTableComponent::CK_FunctionPointer:
      MD = Component.getFunctionDecl();
      break;
    case VTableComponent::CK_CompleteDtorPointer:
      MD = Component.getDestructorDecl();
      break;
    case VTableComponent::CK_DeletingDtorPointer:
      // We've already added the thunk when we saw the complete dtor pointer.
      continue;
    }

    if (MD->getParent() == MostDerivedClass)
      AddThunk(MD, Thunk);
  }
}

ReturnAdjustment VTableBuilder::ComputeReturnAdjustment(BaseOffset Offset) {
  ReturnAdjustment Adjustment;
  
  if (!Offset.isEmpty()) {
    if (Offset.VirtualBase) {
      // Get the virtual base offset offset.
      if (Offset.DerivedClass == MostDerivedClass) {
        // We can get the offset offset directly from our map.
        Adjustment.VBaseOffsetOffset = 
          VBaseOffsetOffsets.lookup(Offset.VirtualBase).getQuantity();
      } else {
        Adjustment.VBaseOffsetOffset = 
          VTables.getVirtualBaseOffsetOffset(Offset.DerivedClass,
                                             Offset.VirtualBase).getQuantity();
      }
    }

    Adjustment.NonVirtual = Offset.NonVirtualOffset.getQuantity();
  }
  
  return Adjustment;
}

BaseOffset
VTableBuilder::ComputeThisAdjustmentBaseOffset(BaseSubobject Base,
                                               BaseSubobject Derived) const {
  const CXXRecordDecl *BaseRD = Base.getBase();
  const CXXRecordDecl *DerivedRD = Derived.getBase();
  
  CXXBasePaths Paths(/*FindAmbiguities=*/true,
                     /*RecordPaths=*/true, /*DetectVirtual=*/true);

  if (!const_cast<CXXRecordDecl *>(DerivedRD)->
      isDerivedFrom(const_cast<CXXRecordDecl *>(BaseRD), Paths)) {
    llvm_unreachable("Class must be derived from the passed in base class!");
  }

  // We have to go through all the paths, and see which one leads us to the
  // right base subobject.
  for (CXXBasePaths::const_paths_iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    BaseOffset Offset = ComputeBaseOffset(Context, DerivedRD, *I);
    
    CharUnits OffsetToBaseSubobject = Offset.NonVirtualOffset;
    
    if (Offset.VirtualBase) {
      // If we have a virtual base class, the non-virtual offset is relative
      // to the virtual base class offset.
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      
      /// Get the virtual base offset, relative to the most derived class 
      /// layout.
      OffsetToBaseSubobject += 
        LayoutClassLayout.getVBaseClassOffset(Offset.VirtualBase);
    } else {
      // Otherwise, the non-virtual offset is relative to the derived class 
      // offset.
      OffsetToBaseSubobject += Derived.getBaseOffset();
    }
    
    // Check if this path gives us the right base subobject.
    if (OffsetToBaseSubobject == Base.getBaseOffset()) {
      // Since we're going from the base class _to_ the derived class, we'll
      // invert the non-virtual offset here.
      Offset.NonVirtualOffset = -Offset.NonVirtualOffset;
      return Offset;
    }      
  }
  
  return BaseOffset();
}
  
ThisAdjustment 
VTableBuilder::ComputeThisAdjustment(const CXXMethodDecl *MD, 
                                     CharUnits BaseOffsetInLayoutClass,
                                     FinalOverriders::OverriderInfo Overrider) {
  // Ignore adjustments for pure virtual member functions.
  if (Overrider.Method->isPure())
    return ThisAdjustment();
  
  BaseSubobject OverriddenBaseSubobject(MD->getParent(), 
                                        BaseOffsetInLayoutClass);
  
  BaseSubobject OverriderBaseSubobject(Overrider.Method->getParent(),
                                       Overrider.Offset);
  
  // Compute the adjustment offset.
  BaseOffset Offset = ComputeThisAdjustmentBaseOffset(OverriddenBaseSubobject,
                                                      OverriderBaseSubobject);
  if (Offset.isEmpty())
    return ThisAdjustment();

  ThisAdjustment Adjustment;
  
  if (Offset.VirtualBase) {
    // Get the vcall offset map for this virtual base.
    VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Offset.VirtualBase];

    if (VCallOffsets.empty()) {
      // We don't have vcall offsets for this virtual base, go ahead and
      // build them.
      VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, MostDerivedClass,
                                         /*FinalOverriders=*/0,
                                         BaseSubobject(Offset.VirtualBase,
                                                       CharUnits::Zero()),
                                         /*BaseIsVirtual=*/true,
                                         /*OffsetInLayoutClass=*/
                                             CharUnits::Zero());
        
      VCallOffsets = Builder.getVCallOffsets();
    }
      
    Adjustment.VCallOffsetOffset = 
      VCallOffsets.getVCallOffsetOffset(MD).getQuantity();
  }

  // Set the non-virtual part of the adjustment.
  Adjustment.NonVirtual = Offset.NonVirtualOffset.getQuantity();
  
  return Adjustment;
}
  
void 
VTableBuilder::AddMethod(const CXXMethodDecl *MD,
                         ReturnAdjustment ReturnAdjustment) {
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    assert(ReturnAdjustment.isEmpty() && 
           "Destructor can't have return adjustment!");

    // Add both the complete destructor and the deleting destructor.
    Components.push_back(VTableComponent::MakeCompleteDtor(DD));
    Components.push_back(VTableComponent::MakeDeletingDtor(DD));
  } else {
    // Add the return adjustment if necessary.
    if (!ReturnAdjustment.isEmpty())
      VTableThunks[Components.size()].Return = ReturnAdjustment;

    // Add the function.
    Components.push_back(VTableComponent::MakeFunction(MD));
  }
}

/// OverridesIndirectMethodInBase - Return whether the given member function
/// overrides any methods in the set of given bases. 
/// Unlike OverridesMethodInBase, this checks "overriders of overriders".
/// For example, if we have:
///
/// struct A { virtual void f(); }
/// struct B : A { virtual void f(); }
/// struct C : B { virtual void f(); }
///
/// OverridesIndirectMethodInBase will return true if given C::f as the method 
/// and { A } as the set of bases.
static bool
OverridesIndirectMethodInBases(const CXXMethodDecl *MD,
                               VTableBuilder::PrimaryBasesSetVectorTy &Bases) {
  if (Bases.count(MD->getParent()))
    return true;
  
  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    
    // Check "indirect overriders".
    if (OverridesIndirectMethodInBases(OverriddenMD, Bases))
      return true;
  }
   
  return false;
}

bool 
VTableBuilder::IsOverriderUsed(const CXXMethodDecl *Overrider,
                               CharUnits BaseOffsetInLayoutClass,
                               const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                               CharUnits FirstBaseOffsetInLayoutClass) const {
  // If the base and the first base in the primary base chain have the same
  // offsets, then this overrider will be used.
  if (BaseOffsetInLayoutClass == FirstBaseOffsetInLayoutClass)
   return true;

  // We know now that Base (or a direct or indirect base of it) is a primary
  // base in part of the class hierarchy, but not a primary base in the most 
  // derived class.
  
  // If the overrider is the first base in the primary base chain, we know
  // that the overrider will be used.
  if (Overrider->getParent() == FirstBaseInPrimaryBaseChain)
    return true;
  
  VTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;

  const CXXRecordDecl *RD = FirstBaseInPrimaryBaseChain;
  PrimaryBases.insert(RD);

  // Now traverse the base chain, starting with the first base, until we find
  // the base that is no longer a primary base.
  while (true) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.isPrimaryBaseVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should always be at offset 0!");

      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      // Now check if this is the primary base that is not a primary base in the
      // most derived class.
      if (LayoutClassLayout.getVBaseClassOffset(PrimaryBase) !=
          FirstBaseOffsetInLayoutClass) {
        // We found it, stop walking the chain.
        break;
      }
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should always be at offset 0!");
    }
    
    if (!PrimaryBases.insert(PrimaryBase))
      llvm_unreachable("Found a duplicate primary base!");

    RD = PrimaryBase;
  }
  
  // If the final overrider is an override of one of the primary bases,
  // then we know that it will be used.
  return OverridesIndirectMethodInBases(Overrider, PrimaryBases);
}

/// FindNearestOverriddenMethod - Given a method, returns the overridden method
/// from the nearest base. Returns null if no method was found.
static const CXXMethodDecl * 
FindNearestOverriddenMethod(const CXXMethodDecl *MD,
                            VTableBuilder::PrimaryBasesSetVectorTy &Bases) {
  OverriddenMethodsSetTy OverriddenMethods;
  ComputeAllOverriddenMethods(MD, OverriddenMethods);
  
  for (int I = Bases.size(), E = 0; I != E; --I) {
    const CXXRecordDecl *PrimaryBase = Bases[I - 1];

    // Now check the overriden methods.
    for (OverriddenMethodsSetTy::const_iterator I = OverriddenMethods.begin(),
         E = OverriddenMethods.end(); I != E; ++I) {
      const CXXMethodDecl *OverriddenMD = *I;
      
      // We found our overridden method.
      if (OverriddenMD->getParent() == PrimaryBase)
        return OverriddenMD;
    }
  }
  
  return 0;
}  

void
VTableBuilder::AddMethods(BaseSubobject Base, CharUnits BaseOffsetInLayoutClass,
                          const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                          CharUnits FirstBaseOffsetInLayoutClass,
                          PrimaryBasesSetVectorTy &PrimaryBases) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    CharUnits PrimaryBaseOffset;
    CharUnits PrimaryBaseOffsetInLayoutClass;
    if (Layout.isPrimaryBaseVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      PrimaryBaseOffsetInLayoutClass =
        LayoutClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
      PrimaryBaseOffsetInLayoutClass = BaseOffsetInLayoutClass;
    }

    AddMethods(BaseSubobject(PrimaryBase, PrimaryBaseOffset),
               PrimaryBaseOffsetInLayoutClass, FirstBaseInPrimaryBaseChain, 
               FirstBaseOffsetInLayoutClass, PrimaryBases);
    
    if (!PrimaryBases.insert(PrimaryBase))
      llvm_unreachable("Found a duplicate primary base!");
  }

  // Now go through all virtual member functions and add them.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
  
    if (!MD->isVirtual())
      continue;

    // Get the final overrider.
    FinalOverriders::OverriderInfo Overrider = 
      Overriders.getOverrider(MD, Base.getBaseOffset());

    // Check if this virtual member function overrides a method in a primary
    // base. If this is the case, and the return type doesn't require adjustment
    // then we can just use the member function from the primary base.
    if (const CXXMethodDecl *OverriddenMD = 
          FindNearestOverriddenMethod(MD, PrimaryBases)) {
      if (ComputeReturnAdjustmentBaseOffset(Context, MD, 
                                            OverriddenMD).isEmpty()) {
        // Replace the method info of the overridden method with our own
        // method.
        assert(MethodInfoMap.count(OverriddenMD) && 
               "Did not find the overridden method!");
        MethodInfo &OverriddenMethodInfo = MethodInfoMap[OverriddenMD];
        
        MethodInfo MethodInfo(Base.getBaseOffset(), BaseOffsetInLayoutClass,
                              OverriddenMethodInfo.VTableIndex);

        assert(!MethodInfoMap.count(MD) &&
               "Should not have method info for this method yet!");
        
        MethodInfoMap.insert(std::make_pair(MD, MethodInfo));
        MethodInfoMap.erase(OverriddenMD);
        
        // If the overridden method exists in a virtual base class or a direct
        // or indirect base class of a virtual base class, we need to emit a
        // thunk if we ever have a class hierarchy where the base class is not
        // a primary base in the complete object.
        if (!isBuildingConstructorVTable() && OverriddenMD != MD) {
          // Compute the this adjustment.
          ThisAdjustment ThisAdjustment =
            ComputeThisAdjustment(OverriddenMD, BaseOffsetInLayoutClass,
                                  Overrider);

          if (ThisAdjustment.VCallOffsetOffset &&
              Overrider.Method->getParent() == MostDerivedClass) {

            // There's no return adjustment from OverriddenMD and MD,
            // but that doesn't mean there isn't one between MD and
            // the final overrider.
            BaseOffset ReturnAdjustmentOffset =
              ComputeReturnAdjustmentBaseOffset(Context, Overrider.Method, MD);
            ReturnAdjustment ReturnAdjustment = 
              ComputeReturnAdjustment(ReturnAdjustmentOffset);

            // This is a virtual thunk for the most derived class, add it.
            AddThunk(Overrider.Method, 
                     ThunkInfo(ThisAdjustment, ReturnAdjustment));
          }
        }

        continue;
      }
    }

    // Insert the method info for this method.
    MethodInfo MethodInfo(Base.getBaseOffset(), BaseOffsetInLayoutClass,
                          Components.size());

    assert(!MethodInfoMap.count(MD) &&
           "Should not have method info for this method yet!");
    MethodInfoMap.insert(std::make_pair(MD, MethodInfo));

    // Check if this overrider is going to be used.
    const CXXMethodDecl *OverriderMD = Overrider.Method;
    if (!IsOverriderUsed(OverriderMD, BaseOffsetInLayoutClass,
                         FirstBaseInPrimaryBaseChain, 
                         FirstBaseOffsetInLayoutClass)) {
      Components.push_back(VTableComponent::MakeUnusedFunction(OverriderMD));
      continue;
    }
    
    // Check if this overrider needs a return adjustment.
    // We don't want to do this for pure virtual member functions.
    BaseOffset ReturnAdjustmentOffset;
    if (!OverriderMD->isPure()) {
      ReturnAdjustmentOffset = 
        ComputeReturnAdjustmentBaseOffset(Context, OverriderMD, MD);
    }

    ReturnAdjustment ReturnAdjustment = 
      ComputeReturnAdjustment(ReturnAdjustmentOffset);
    
    AddMethod(Overrider.Method, ReturnAdjustment);
  }
}

void VTableBuilder::LayoutVTable() {
  LayoutPrimaryAndSecondaryVTables(BaseSubobject(MostDerivedClass,
                                                 CharUnits::Zero()),
                                   /*BaseIsMorallyVirtual=*/false,
                                   MostDerivedClassIsVirtual,
                                   MostDerivedClassOffset);
  
  VisitedVirtualBasesSetTy VBases;
  
  // Determine the primary virtual bases.
  DeterminePrimaryVirtualBases(MostDerivedClass, MostDerivedClassOffset, 
                               VBases);
  VBases.clear();
  
  LayoutVTablesForVirtualBases(MostDerivedClass, VBases);

  // -fapple-kext adds an extra entry at end of vtbl.
  bool IsAppleKext = Context.getLangOpts().AppleKext;
  if (IsAppleKext)
    Components.push_back(VTableComponent::MakeVCallOffset(CharUnits::Zero()));
}
  
void
VTableBuilder::LayoutPrimaryAndSecondaryVTables(BaseSubobject Base,
                                                bool BaseIsMorallyVirtual,
                                                bool BaseIsVirtualInLayoutClass,
                                                CharUnits OffsetInLayoutClass) {
  assert(Base.getBase()->isDynamicClass() && "class does not have a vtable!");

  // Add vcall and vbase offsets for this vtable.
  VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, LayoutClass, &Overriders,
                                     Base, BaseIsVirtualInLayoutClass, 
                                     OffsetInLayoutClass);
  Components.append(Builder.components_begin(), Builder.components_end());
  
  // Check if we need to add these vcall offsets.
  if (BaseIsVirtualInLayoutClass && !Builder.getVCallOffsets().empty()) {
    VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Base.getBase()];
    
    if (VCallOffsets.empty())
      VCallOffsets = Builder.getVCallOffsets();
  }

  // If we're laying out the most derived class we want to keep track of the
  // virtual base class offset offsets.
  if (Base.getBase() == MostDerivedClass)
    VBaseOffsetOffsets = Builder.getVBaseOffsetOffsets();

  // Add the offset to top.
  CharUnits OffsetToTop = MostDerivedClassOffset - OffsetInLayoutClass;
  Components.push_back(
    VTableComponent::MakeOffsetToTop(OffsetToTop));
  
  // Next, add the RTTI.
  Components.push_back(VTableComponent::MakeRTTI(MostDerivedClass));
  
  uint64_t AddressPoint = Components.size();

  // Now go through all virtual member functions and add them.
  PrimaryBasesSetVectorTy PrimaryBases;
  AddMethods(Base, OffsetInLayoutClass,
             Base.getBase(), OffsetInLayoutClass, 
             PrimaryBases);

  // Compute 'this' pointer adjustments.
  ComputeThisAdjustments();

  // Add all address points.
  const CXXRecordDecl *RD = Base.getBase();
  while (true) {
    AddressPoints.insert(std::make_pair(
      BaseSubobject(RD, OffsetInLayoutClass),
      AddressPoint));

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.isPrimaryBaseVirtual()) {
      // Check if this virtual primary base is a primary base in the layout
      // class. If it's not, we don't want to add it.
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      if (LayoutClassLayout.getVBaseClassOffset(PrimaryBase) !=
          OffsetInLayoutClass) {
        // We don't want to add this class (or any of its primary bases).
        break;
      }
    }

    RD = PrimaryBase;
  }

  // Layout secondary vtables.
  LayoutSecondaryVTables(Base, BaseIsMorallyVirtual, OffsetInLayoutClass);
}

void VTableBuilder::LayoutSecondaryVTables(BaseSubobject Base,
                                           bool BaseIsMorallyVirtual,
                                           CharUnits OffsetInLayoutClass) {
  // Itanium C++ ABI 2.5.2:
  //   Following the primary virtual table of a derived class are secondary 
  //   virtual tables for each of its proper base classes, except any primary
  //   base(s) with which it shares its primary virtual table.

  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    // Ignore virtual bases, we'll emit them later.
    if (I->isVirtual())
      continue;
    
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Ignore bases that don't have a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    if (isBuildingConstructorVTable()) {
      // Itanium C++ ABI 2.6.4:
      //   Some of the base class subobjects may not need construction virtual
      //   tables, which will therefore not be present in the construction
      //   virtual table group, even though the subobject virtual tables are
      //   present in the main virtual table group for the complete object.
      if (!BaseIsMorallyVirtual && !BaseDecl->getNumVBases())
        continue;
    }

    // Get the base offset of this base.
    CharUnits RelativeBaseOffset = Layout.getBaseClassOffset(BaseDecl);
    CharUnits BaseOffset = Base.getBaseOffset() + RelativeBaseOffset;
    
    CharUnits BaseOffsetInLayoutClass = 
      OffsetInLayoutClass + RelativeBaseOffset;
    
    // Don't emit a secondary vtable for a primary base. We might however want 
    // to emit secondary vtables for other bases of this base.
    if (BaseDecl == PrimaryBase) {
      LayoutSecondaryVTables(BaseSubobject(BaseDecl, BaseOffset),
                             BaseIsMorallyVirtual, BaseOffsetInLayoutClass);
      continue;
    }

    // Layout the primary vtable (and any secondary vtables) for this base.
    LayoutPrimaryAndSecondaryVTables(
      BaseSubobject(BaseDecl, BaseOffset),
      BaseIsMorallyVirtual,
      /*BaseIsVirtualInLayoutClass=*/false,
      BaseOffsetInLayoutClass);
  }
}

void
VTableBuilder::DeterminePrimaryVirtualBases(const CXXRecordDecl *RD,
                                            CharUnits OffsetInLayoutClass,
                                            VisitedVirtualBasesSetTy &VBases) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  // Check if this base has a primary base.
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {

    // Check if it's virtual.
    if (Layout.isPrimaryBaseVirtual()) {
      bool IsPrimaryVirtualBase = true;

      if (isBuildingConstructorVTable()) {
        // Check if the base is actually a primary base in the class we use for
        // layout.
        const ASTRecordLayout &LayoutClassLayout =
          Context.getASTRecordLayout(LayoutClass);

        CharUnits PrimaryBaseOffsetInLayoutClass =
          LayoutClassLayout.getVBaseClassOffset(PrimaryBase);
        
        // We know that the base is not a primary base in the layout class if 
        // the base offsets are different.
        if (PrimaryBaseOffsetInLayoutClass != OffsetInLayoutClass)
          IsPrimaryVirtualBase = false;
      }
        
      if (IsPrimaryVirtualBase)
        PrimaryVirtualBases.insert(PrimaryBase);
    }
  }

  // Traverse bases, looking for more primary virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffsetInLayoutClass;
    
    if (I->isVirtual()) {
      if (!VBases.insert(BaseDecl))
        continue;
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffsetInLayoutClass = 
        OffsetInLayoutClass + Layout.getBaseClassOffset(BaseDecl);
    }

    DeterminePrimaryVirtualBases(BaseDecl, BaseOffsetInLayoutClass, VBases);
  }
}

void
VTableBuilder::LayoutVTablesForVirtualBases(const CXXRecordDecl *RD, 
                                            VisitedVirtualBasesSetTy &VBases) {
  // Itanium C++ ABI 2.5.2:
  //   Then come the virtual base virtual tables, also in inheritance graph
  //   order, and again excluding primary bases (which share virtual tables with
  //   the classes for which they are primary).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this base needs a vtable. (If it's virtual, not a primary base
    // of some other class, and we haven't visited it before).
    if (I->isVirtual() && BaseDecl->isDynamicClass() && 
        !PrimaryVirtualBases.count(BaseDecl) && VBases.insert(BaseDecl)) {
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      CharUnits BaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      CharUnits BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);

      LayoutPrimaryAndSecondaryVTables(
        BaseSubobject(BaseDecl, BaseOffset),
        /*BaseIsMorallyVirtual=*/true,
        /*BaseIsVirtualInLayoutClass=*/true,
        BaseOffsetInLayoutClass);
    }
    
    // We only need to check the base for virtual base vtables if it actually
    // has virtual bases.
    if (BaseDecl->getNumVBases())
      LayoutVTablesForVirtualBases(BaseDecl, VBases);
  }
}

/// dumpLayout - Dump the vtable layout.
void VTableBuilder::dumpLayout(raw_ostream& Out) {

  if (isBuildingConstructorVTable()) {
    Out << "Construction vtable for ('";
    Out << MostDerivedClass->getQualifiedNameAsString() << "', ";
    Out << MostDerivedClassOffset.getQuantity() << ") in '";
    Out << LayoutClass->getQualifiedNameAsString();
  } else {
    Out << "Vtable for '";
    Out << MostDerivedClass->getQualifiedNameAsString();
  }
  Out << "' (" << Components.size() << " entries).\n";

  // Iterate through the address points and insert them into a new map where
  // they are keyed by the index and not the base object.
  // Since an address point can be shared by multiple subobjects, we use an
  // STL multimap.
  std::multimap<uint64_t, BaseSubobject> AddressPointsByIndex;
  for (AddressPointsMapTy::const_iterator I = AddressPoints.begin(), 
       E = AddressPoints.end(); I != E; ++I) {
    const BaseSubobject& Base = I->first;
    uint64_t Index = I->second;
    
    AddressPointsByIndex.insert(std::make_pair(Index, Base));
  }
  
  for (unsigned I = 0, E = Components.size(); I != E; ++I) {
    uint64_t Index = I;

    Out << llvm::format("%4d | ", I);

    const VTableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {

    case VTableComponent::CK_VCallOffset:
      Out << "vcall_offset ("
          << Component.getVCallOffset().getQuantity() 
          << ")";
      break;

    case VTableComponent::CK_VBaseOffset:
      Out << "vbase_offset ("
          << Component.getVBaseOffset().getQuantity()
          << ")";
      break;

    case VTableComponent::CK_OffsetToTop:
      Out << "offset_to_top ("
          << Component.getOffsetToTop().getQuantity()
          << ")";
      break;
    
    case VTableComponent::CK_RTTI:
      Out << Component.getRTTIDecl()->getQualifiedNameAsString() << " RTTI";
      break;
    
    case VTableComponent::CK_FunctionPointer: {
      const CXXMethodDecl *MD = Component.getFunctionDecl();

      std::string Str = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual, 
                                    MD);
      Out << Str;
      if (MD->isPure())
        Out << " [pure]";

      if (MD->isDeleted())
        Out << " [deleted]";

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this function pointer has a return adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "\n       [return adjustment: ";
          Out << Thunk.Return.NonVirtual << " non-virtual";
          
          if (Thunk.Return.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          Out << ']';
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }

          Out << ']';
        }          
      }

      break;
    }

    case VTableComponent::CK_CompleteDtorPointer: 
    case VTableComponent::CK_DeletingDtorPointer: {
      bool IsComplete = 
        Component.getKind() == VTableComponent::CK_CompleteDtorPointer;
      
      const CXXDestructorDecl *DD = Component.getDestructorDecl();
      
      Out << DD->getQualifiedNameAsString();
      if (IsComplete)
        Out << "() [complete]";
      else
        Out << "() [deleting]";

      if (DD->isPure())
        Out << " [pure]";

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this destructor has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
          
          Out << ']';
        }          
      }        

      break;
    }

    case VTableComponent::CK_UnusedFunctionPointer: {
      const CXXMethodDecl *MD = Component.getUnusedFunctionDecl();

      std::string Str = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual, 
                                    MD);
      Out << "[unused] " << Str;
      if (MD->isPure())
        Out << " [pure]";
    }

    }

    Out << '\n';
    
    // Dump the next address point.
    uint64_t NextIndex = Index + 1;
    if (AddressPointsByIndex.count(NextIndex)) {
      if (AddressPointsByIndex.count(NextIndex) == 1) {
        const BaseSubobject &Base = 
          AddressPointsByIndex.find(NextIndex)->second;
        
        Out << "       -- (" << Base.getBase()->getQualifiedNameAsString();
        Out << ", " << Base.getBaseOffset().getQuantity();
        Out << ") vtable address --\n";
      } else {
        CharUnits BaseOffset =
          AddressPointsByIndex.lower_bound(NextIndex)->second.getBaseOffset();
        
        // We store the class names in a set to get a stable order.
        std::set<std::string> ClassNames;
        for (std::multimap<uint64_t, BaseSubobject>::const_iterator I =
             AddressPointsByIndex.lower_bound(NextIndex), E =
             AddressPointsByIndex.upper_bound(NextIndex); I != E; ++I) {
          assert(I->second.getBaseOffset() == BaseOffset &&
                 "Invalid base offset!");
          const CXXRecordDecl *RD = I->second.getBase();
          ClassNames.insert(RD->getQualifiedNameAsString());
        }
        
        for (std::set<std::string>::const_iterator I = ClassNames.begin(),
             E = ClassNames.end(); I != E; ++I) {
          Out << "       -- (" << *I;
          Out << ", " << BaseOffset.getQuantity() << ") vtable address --\n";
        }
      }
    }
  }

  Out << '\n';
  
  if (isBuildingConstructorVTable())
    return;
  
  if (MostDerivedClass->getNumVBases()) {
    // We store the virtual base class names and their offsets in a map to get
    // a stable order.

    std::map<std::string, CharUnits> ClassNamesAndOffsets;
    for (VBaseOffsetOffsetsMapTy::const_iterator I = VBaseOffsetOffsets.begin(),
         E = VBaseOffsetOffsets.end(); I != E; ++I) {
      std::string ClassName = I->first->getQualifiedNameAsString();
      CharUnits OffsetOffset = I->second;
      ClassNamesAndOffsets.insert(
          std::make_pair(ClassName, OffsetOffset));
    }
    
    Out << "Virtual base offset offsets for '";
    Out << MostDerivedClass->getQualifiedNameAsString() << "' (";
    Out << ClassNamesAndOffsets.size();
    Out << (ClassNamesAndOffsets.size() == 1 ? " entry" : " entries") << ").\n";

    for (std::map<std::string, CharUnits>::const_iterator I =
         ClassNamesAndOffsets.begin(), E = ClassNamesAndOffsets.end(); 
         I != E; ++I)
      Out << "   " << I->first << " | " << I->second.getQuantity() << '\n';

    Out << "\n";
  }
  
  if (!Thunks.empty()) {
    // We store the method names in a map to get a stable order.
    std::map<std::string, const CXXMethodDecl *> MethodNamesAndDecls;
    
    for (ThunksMapTy::const_iterator I = Thunks.begin(), E = Thunks.end();
         I != E; ++I) {
      const CXXMethodDecl *MD = I->first;
      std::string MethodName = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual,
                                    MD);
      
      MethodNamesAndDecls.insert(std::make_pair(MethodName, MD));
    }

    for (std::map<std::string, const CXXMethodDecl *>::const_iterator I =
         MethodNamesAndDecls.begin(), E = MethodNamesAndDecls.end(); 
         I != E; ++I) {
      const std::string &MethodName = I->first;
      const CXXMethodDecl *MD = I->second;

      ThunkInfoVectorTy ThunksVector = Thunks[MD];
      std::sort(ThunksVector.begin(), ThunksVector.end());

      Out << "Thunks for '" << MethodName << "' (" << ThunksVector.size();
      Out << (ThunksVector.size() == 1 ? " entry" : " entries") << ").\n";
      
      for (unsigned I = 0, E = ThunksVector.size(); I != E; ++I) {
        const ThunkInfo &Thunk = ThunksVector[I];

        Out << llvm::format("%4d | ", I);
        
        // If this function pointer has a return pointer adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "return adjustment: " << Thunk.This.NonVirtual;
          Out << " non-virtual";
          if (Thunk.Return.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          if (!Thunk.This.isEmpty())
            Out << "\n       ";
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
        }
        
        Out << '\n';
      }
      
      Out << '\n';
    }
  }

  // Compute the vtable indices for all the member functions.
  // Store them in a map keyed by the index so we'll get a sorted table.
  std::map<uint64_t, std::string> IndicesMap;

  for (CXXRecordDecl::method_iterator i = MostDerivedClass->method_begin(),
       e = MostDerivedClass->method_end(); i != e; ++i) {
    const CXXMethodDecl *MD = *i;
    
    // We only want virtual member functions.
    if (!MD->isVirtual())
      continue;

    std::string MethodName =
      PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual,
                                  MD);

    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      IndicesMap[VTables.getMethodVTableIndex(GlobalDecl(DD, Dtor_Complete))] =
        MethodName + " [complete]";
      IndicesMap[VTables.getMethodVTableIndex(GlobalDecl(DD, Dtor_Deleting))] =
        MethodName + " [deleting]";
    } else {
      IndicesMap[VTables.getMethodVTableIndex(MD)] = MethodName;
    }
  }

  // Print the vtable indices for all the member functions.
  if (!IndicesMap.empty()) {
    Out << "VTable indices for '";
    Out << MostDerivedClass->getQualifiedNameAsString();
    Out << "' (" << IndicesMap.size() << " entries).\n";

    for (std::map<uint64_t, std::string>::const_iterator I = IndicesMap.begin(),
         E = IndicesMap.end(); I != E; ++I) {
      uint64_t VTableIndex = I->first;
      const std::string &MethodName = I->second;

      Out << llvm::format(" %4" PRIu64 " | ", VTableIndex) << MethodName
          << '\n';
    }
  }

  Out << '\n';
}
  
}

VTableLayout::VTableLayout(uint64_t NumVTableComponents,
                           const VTableComponent *VTableComponents,
                           uint64_t NumVTableThunks,
                           const VTableThunkTy *VTableThunks,
                           const AddressPointsMapTy &AddressPoints)
  : NumVTableComponents(NumVTableComponents),
    VTableComponents(new VTableComponent[NumVTableComponents]),
    NumVTableThunks(NumVTableThunks),
    VTableThunks(new VTableThunkTy[NumVTableThunks]),
    AddressPoints(AddressPoints) {
  std::copy(VTableComponents, VTableComponents+NumVTableComponents,
            this->VTableComponents.get());
  std::copy(VTableThunks, VTableThunks+NumVTableThunks,
            this->VTableThunks.get());
}

VTableLayout::~VTableLayout() { }

VTableContext::~VTableContext() {
  llvm::DeleteContainerSeconds(VTableLayouts);
}

static void 
CollectPrimaryBases(const CXXRecordDecl *RD, ASTContext &Context,
                    VTableBuilder::PrimaryBasesSetVectorTy &PrimaryBases) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  if (!PrimaryBase)
    return;

  CollectPrimaryBases(PrimaryBase, Context, PrimaryBases);

  if (!PrimaryBases.insert(PrimaryBase))
    llvm_unreachable("Found a duplicate primary base!");
}

void VTableContext::ComputeMethodVTableIndices(const CXXRecordDecl *RD) {
  
  // Itanium C++ ABI 2.5.2:
  //   The order of the virtual function pointers in a virtual table is the 
  //   order of declaration of the corresponding member functions in the class.
  //
  //   There is an entry for any virtual function declared in a class, 
  //   whether it is a new function or overrides a base class function, 
  //   unless it overrides a function from the primary base, and conversion
  //   between their return types does not require an adjustment. 

  int64_t CurrentIndex = 0;
  
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  
  if (PrimaryBase) {
    assert(PrimaryBase->isCompleteDefinition() && 
           "Should have the definition decl of the primary base!");

    // Since the record decl shares its vtable pointer with the primary base
    // we need to start counting at the end of the primary base's vtable.
    CurrentIndex = getNumVirtualFunctionPointers(PrimaryBase);
  }

  // Collect all the primary bases, so we can check whether methods override
  // a method from the base.
  VTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;
  CollectPrimaryBases(RD, Context, PrimaryBases);

  const CXXDestructorDecl *ImplicitVirtualDtor = 0;
  
  for (CXXRecordDecl::method_iterator i = RD->method_begin(),
       e = RD->method_end(); i != e; ++i) {
    const CXXMethodDecl *MD = *i;

    // We only want virtual methods.
    if (!MD->isVirtual())
      continue;

    // Check if this method overrides a method in the primary base.
    if (const CXXMethodDecl *OverriddenMD = 
          FindNearestOverriddenMethod(MD, PrimaryBases)) {
      // Check if converting from the return type of the method to the 
      // return type of the overridden method requires conversion.
      if (ComputeReturnAdjustmentBaseOffset(Context, MD, 
                                            OverriddenMD).isEmpty()) {
        // This index is shared between the index in the vtable of the primary
        // base class.
        if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
          const CXXDestructorDecl *OverriddenDD = 
            cast<CXXDestructorDecl>(OverriddenMD);
          
          // Add both the complete and deleting entries.
          MethodVTableIndices[GlobalDecl(DD, Dtor_Complete)] = 
            getMethodVTableIndex(GlobalDecl(OverriddenDD, Dtor_Complete));
          MethodVTableIndices[GlobalDecl(DD, Dtor_Deleting)] = 
            getMethodVTableIndex(GlobalDecl(OverriddenDD, Dtor_Deleting));
        } else {
          MethodVTableIndices[MD] = getMethodVTableIndex(OverriddenMD);
        }
        
        // We don't need to add an entry for this method.
        continue;
      }
    }
    
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      if (MD->isImplicit()) {
        assert(!ImplicitVirtualDtor && 
               "Did already see an implicit virtual dtor!");
        ImplicitVirtualDtor = DD;
        continue;
      } 

      // Add the complete dtor.
      MethodVTableIndices[GlobalDecl(DD, Dtor_Complete)] = CurrentIndex++;
      
      // Add the deleting dtor.
      MethodVTableIndices[GlobalDecl(DD, Dtor_Deleting)] = CurrentIndex++;
    } else {
      // Add the entry.
      MethodVTableIndices[MD] = CurrentIndex++;
    }
  }

  if (ImplicitVirtualDtor) {
    // Itanium C++ ABI 2.5.2:
    //   If a class has an implicitly-defined virtual destructor, 
    //   its entries come after the declared virtual function pointers.

    // Add the complete dtor.
    MethodVTableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Complete)] = 
      CurrentIndex++;
    
    // Add the deleting dtor.
    MethodVTableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Deleting)] = 
      CurrentIndex++;
  }
  
  NumVirtualFunctionPointers[RD] = CurrentIndex;
}

uint64_t VTableContext::getNumVirtualFunctionPointers(const CXXRecordDecl *RD) {
  llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I = 
    NumVirtualFunctionPointers.find(RD);
  if (I != NumVirtualFunctionPointers.end())
    return I->second;

  ComputeMethodVTableIndices(RD);

  I = NumVirtualFunctionPointers.find(RD);
  assert(I != NumVirtualFunctionPointers.end() && "Did not find entry!");
  return I->second;
}
      
uint64_t VTableContext::getMethodVTableIndex(GlobalDecl GD) {
  MethodVTableIndicesTy::iterator I = MethodVTableIndices.find(GD);
  if (I != MethodVTableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  ComputeMethodVTableIndices(RD);

  I = MethodVTableIndices.find(GD);
  assert(I != MethodVTableIndices.end() && "Did not find index!");
  return I->second;
}

CharUnits 
VTableContext::getVirtualBaseOffsetOffset(const CXXRecordDecl *RD, 
                                          const CXXRecordDecl *VBase) {
  ClassPairTy ClassPair(RD, VBase);
  
  VirtualBaseClassOffsetOffsetsMapTy::iterator I = 
    VirtualBaseClassOffsetOffsets.find(ClassPair);
  if (I != VirtualBaseClassOffsetOffsets.end())
    return I->second;
  
  VCallAndVBaseOffsetBuilder Builder(RD, RD, /*FinalOverriders=*/0,
                                     BaseSubobject(RD, CharUnits::Zero()),
                                     /*BaseIsVirtual=*/false,
                                     /*OffsetInLayoutClass=*/CharUnits::Zero());

  for (VCallAndVBaseOffsetBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
       Builder.getVBaseOffsetOffsets().begin(), 
       E = Builder.getVBaseOffsetOffsets().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(
        std::make_pair(ClassPair, I->second));
  }
  
  I = VirtualBaseClassOffsetOffsets.find(ClassPair);
  assert(I != VirtualBaseClassOffsetOffsets.end() && "Did not find index!");
  
  return I->second;
}

static VTableLayout *CreateVTableLayout(const VTableBuilder &Builder) {
  SmallVector<VTableLayout::VTableThunkTy, 1>
    VTableThunks(Builder.vtable_thunks_begin(), Builder.vtable_thunks_end());
  std::sort(VTableThunks.begin(), VTableThunks.end());

  return new VTableLayout(Builder.getNumVTableComponents(),
                          Builder.vtable_component_begin(),
                          VTableThunks.size(),
                          VTableThunks.data(),
                          Builder.getAddressPoints());
}

void VTableContext::ComputeVTableRelatedInformation(const CXXRecordDecl *RD) {
  const VTableLayout *&Entry = VTableLayouts[RD];

  // Check if we've computed this information before.
  if (Entry)
    return;

  VTableBuilder Builder(*this, RD, CharUnits::Zero(), 
                        /*MostDerivedClassIsVirtual=*/0, RD);
  Entry = CreateVTableLayout(Builder);

  // Add the known thunks.
  Thunks.insert(Builder.thunks_begin(), Builder.thunks_end());

  // If we don't have the vbase information for this class, insert it.
  // getVirtualBaseOffsetOffset will compute it separately without computing
  // the rest of the vtable related information.
  if (!RD->getNumVBases())
    return;
  
  const RecordType *VBaseRT = 
    RD->vbases_begin()->getType()->getAs<RecordType>();
  const CXXRecordDecl *VBase = cast<CXXRecordDecl>(VBaseRT->getDecl());
  
  if (VirtualBaseClassOffsetOffsets.count(std::make_pair(RD, VBase)))
    return;
  
  for (VTableBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
       Builder.getVBaseOffsetOffsets().begin(), 
       E = Builder.getVBaseOffsetOffsets().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(std::make_pair(ClassPair, I->second));
  }
}

VTableLayout *VTableContext::createConstructionVTableLayout(
                                          const CXXRecordDecl *MostDerivedClass,
                                          CharUnits MostDerivedClassOffset,
                                          bool MostDerivedClassIsVirtual,
                                          const CXXRecordDecl *LayoutClass) {
  VTableBuilder Builder(*this, MostDerivedClass, MostDerivedClassOffset, 
                        MostDerivedClassIsVirtual, LayoutClass);
  return CreateVTableLayout(Builder);
}
