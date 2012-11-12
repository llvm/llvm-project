//===--- CGVTT.cpp - Emit LLVM Code for C++ VTTs --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of VTTs (vtable tables).
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CGCXXABI.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/VTTBuilder.h"
using namespace clang;
using namespace CodeGen;

static llvm::Constant *
GetAddrOfVTTVTable(CodeGenVTables &CGVT, const CXXRecordDecl *MostDerivedClass,
                   const VTTVTable &VTable,
                   llvm::GlobalVariable::LinkageTypes Linkage,
                   llvm::DenseMap<BaseSubobject, uint64_t> &AddressPoints) {
  if (VTable.getBase() == MostDerivedClass) {
    assert(VTable.getBaseOffset().isZero() &&
           "Most derived class vtable must have a zero offset!");
    // This is a regular vtable.
    return CGVT.GetAddrOfVTable(MostDerivedClass);
  }
  
  return CGVT.GenerateConstructionVTable(MostDerivedClass, 
                                         VTable.getBaseSubobject(),
                                         VTable.isVirtual(),
                                         Linkage,
                                         AddressPoints);
}

void
CodeGenVTables::EmitVTTDefinition(llvm::GlobalVariable *VTT,
                                  llvm::GlobalVariable::LinkageTypes Linkage,
                                  const CXXRecordDecl *RD) {
  VTTBuilder Builder(CGM.getContext(), RD, /*GenerateDefinition=*/true);

  llvm::Type *Int8PtrTy = CGM.Int8PtrTy, *Int64Ty = CGM.Int64Ty;
  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(Int8PtrTy, Builder.getVTTComponents().size());
  
  SmallVector<llvm::Constant *, 8> VTables;
  SmallVector<VTableAddressPointsMapTy, 8> VTableAddressPoints;
  for (const VTTVTable *i = Builder.getVTTVTables().begin(),
                       *e = Builder.getVTTVTables().end(); i != e; ++i) {
    VTableAddressPoints.push_back(VTableAddressPointsMapTy());
    VTables.push_back(GetAddrOfVTTVTable(*this, RD, *i, Linkage,
                                         VTableAddressPoints.back()));
  }

  SmallVector<llvm::Constant *, 8> VTTComponents;
  for (const VTTComponent *i = Builder.getVTTComponents().begin(),
                          *e = Builder.getVTTComponents().end(); i != e; ++i) {
    const VTTVTable &VTTVT = Builder.getVTTVTables()[i->VTableIndex];
    llvm::Constant *VTable = VTables[i->VTableIndex];
    uint64_t AddressPoint;
    if (VTTVT.getBase() == RD) {
      // Just get the address point for the regular vtable.
      AddressPoint = VTContext.getVTableLayout(RD)
                              .getAddressPoint(i->VTableBase);
      assert(AddressPoint != 0 && "Did not find vtable address point!");
    } else {
      AddressPoint = VTableAddressPoints[i->VTableIndex].lookup(i->VTableBase);
      assert(AddressPoint != 0 && "Did not find ctor vtable address point!");
    }

     llvm::Value *Idxs[] = {
       llvm::ConstantInt::get(Int64Ty, 0),
       llvm::ConstantInt::get(Int64Ty, AddressPoint)
     };

     llvm::Constant *Init = 
       llvm::ConstantExpr::getInBoundsGetElementPtr(VTable, Idxs);

     Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);

     VTTComponents.push_back(Init);
  }

  llvm::Constant *Init = llvm::ConstantArray::get(ArrayType, VTTComponents);

  VTT->setInitializer(Init);

  // Set the correct linkage.
  VTT->setLinkage(Linkage);

  // Set the right visibility.
  CGM.setTypeVisibility(VTT, RD, CodeGenModule::TVK_ForVTT);
}

llvm::GlobalVariable *CodeGenVTables::GetAddrOfVTT(const CXXRecordDecl *RD) {
  assert(RD->getNumVBases() && "Only classes with virtual bases need a VTT");

  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXVTT(RD, Out);
  Out.flush();
  StringRef Name = OutName.str();

  // This will also defer the definition of the VTT.
  (void) GetAddrOfVTable(RD);

  VTTBuilder Builder(CGM.getContext(), RD, /*GenerateDefinition=*/false);

  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(CGM.Int8PtrTy, Builder.getVTTComponents().size());

  llvm::GlobalVariable *GV =
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, ArrayType, 
                                          llvm::GlobalValue::ExternalLinkage);
  GV->setUnnamedAddr(true);
  return GV;
}

bool CodeGenVTables::needsVTTParameter(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  
  // We don't have any virtual bases, just return early.
  if (!MD->getParent()->getNumVBases())
    return false;
  
  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return true;
  
  return false;
}

uint64_t CodeGenVTables::getSubVTTIndex(const CXXRecordDecl *RD, 
                                        BaseSubobject Base) {
  BaseSubobjectPairTy ClassSubobjectPair(RD, Base);

  SubVTTIndiciesMapTy::iterator I = SubVTTIndicies.find(ClassSubobjectPair);
  if (I != SubVTTIndicies.end())
    return I->second;
  
  VTTBuilder Builder(CGM.getContext(), RD, /*GenerateDefinition=*/false);

  for (llvm::DenseMap<BaseSubobject, uint64_t>::const_iterator I =
       Builder.getSubVTTIndicies().begin(), 
       E = Builder.getSubVTTIndicies().end(); I != E; ++I) {
    // Insert all indices.
    BaseSubobjectPairTy ClassSubobjectPair(RD, I->first);
    
    SubVTTIndicies.insert(std::make_pair(ClassSubobjectPair, I->second));
  }
    
  I = SubVTTIndicies.find(ClassSubobjectPair);
  assert(I != SubVTTIndicies.end() && "Did not find index!");
  
  return I->second;
}

uint64_t 
CodeGenVTables::getSecondaryVirtualPointerIndex(const CXXRecordDecl *RD,
                                                BaseSubobject Base) {
  SecondaryVirtualPointerIndicesMapTy::iterator I =
    SecondaryVirtualPointerIndices.find(std::make_pair(RD, Base));

  if (I != SecondaryVirtualPointerIndices.end())
    return I->second;

  VTTBuilder Builder(CGM.getContext(), RD, /*GenerateDefinition=*/false);

  // Insert all secondary vpointer indices.
  for (llvm::DenseMap<BaseSubobject, uint64_t>::const_iterator I = 
       Builder.getSecondaryVirtualPointerIndices().begin(),
       E = Builder.getSecondaryVirtualPointerIndices().end(); I != E; ++I) {
    std::pair<const CXXRecordDecl *, BaseSubobject> Pair =
      std::make_pair(RD, I->first);
    
    SecondaryVirtualPointerIndices.insert(std::make_pair(Pair, I->second));
  }

  I = SecondaryVirtualPointerIndices.find(std::make_pair(RD, Base));
  assert(I != SecondaryVirtualPointerIndices.end() && "Did not find index!");
  
  return I->second;
}

