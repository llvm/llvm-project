static bool foldMemChr(CallInst *Call, DomTreeUpdater *DTU,
                       const DataLayout &DL) {
  Value *Ptr = Call->getArgOperand(0);
  Value *Val = Call->getArgOperand(1);
  Value *Len = Call->getArgOperand(2);

  // If length is not a constant, we can't do the optimization
  auto *LenC = dyn_cast<ConstantInt>(Len);
  if (!LenC)
    return false;

  uint64_t Length = LenC->getZExtValue();

  // Check if this is a small memchr we should inline
  if (Length <= MemChrInlineThreshold) {
    IRBuilder<> IRB(Call);

    // Truncate the search value to i8
    Value *ByteVal = IRB.CreateTrunc(Val, IRB.getInt8Ty());

    // Initialize result to null
    Value *Result = ConstantPointerNull::get(cast<PointerType>(Call->getType()));

    // For each byte up to Length
    for (unsigned i = 0; i < Length; i++) {
      Value *CurPtr = i == 0 ? Ptr : 
                     IRB.CreateGEP(IRB.getInt8Ty(), Ptr, 
                                 ConstantInt::get(DL.getIndexType(Call->getType()), i));
      Value *CurByte = IRB.CreateLoad(IRB.getInt8Ty(), CurPtr);
      Value *CmpRes = IRB.CreateICmpEQ(CurByte, ByteVal);
      Result = IRB.CreateSelect(CmpRes, CurPtr, Result);
    }

    // Replace the call with our expanded version
    Call->replaceAllUsesWith(Result);
    Call->eraseFromParent();
    return true;
  }

  return false;
}

// ... rest of the file unchanged ... 