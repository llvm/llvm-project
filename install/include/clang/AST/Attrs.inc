/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute classes' definitions                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_ATTR_CLASSES_INC
#define LLVM_CLANG_ATTR_CLASSES_INC
static inline void DelimitAttributeArgument(raw_ostream& OS, bool& IsFirst) {
  if (IsFirst) {
    IsFirst = false;
    OS << "(";
  } else
    OS << ", ";
}
class CLANG_ABI AArch64SVEPcsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_aarch64_sve_pcs = 0,
    CXX11_clang_aarch64_sve_pcs = 1,
    C23_clang_aarch64_sve_pcs = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AArch64SVEPcsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AArch64SVEPcsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AArch64SVEPcsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_aarch64_sve_pcs);
  static AArch64SVEPcsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_aarch64_sve_pcs);

  // Constructors
  AArch64SVEPcsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AArch64SVEPcsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AArch64SVEPcs; }
};

class CLANG_ABI AArch64VectorPcsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_aarch64_vector_pcs = 0,
    CXX11_clang_aarch64_vector_pcs = 1,
    C23_clang_aarch64_vector_pcs = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AArch64VectorPcsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AArch64VectorPcsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AArch64VectorPcsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_aarch64_vector_pcs);
  static AArch64VectorPcsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_aarch64_vector_pcs);

  // Constructors
  AArch64VectorPcsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AArch64VectorPcsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AArch64VectorPcs; }
};

class CLANG_ABI AMDGPUFlatWorkGroupSizeAttr : public InheritableAttr {
Expr * min;

Expr * max;

public:
  enum Spelling {
    GNU_amdgpu_flat_work_group_size = 0,
    CXX11_clang_amdgpu_flat_work_group_size = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUFlatWorkGroupSizeAttr *CreateImplicit(ASTContext &Ctx, Expr * Min, Expr * Max, const AttributeCommonInfo &CommonInfo);
  static AMDGPUFlatWorkGroupSizeAttr *Create(ASTContext &Ctx, Expr * Min, Expr * Max, const AttributeCommonInfo &CommonInfo);
  static AMDGPUFlatWorkGroupSizeAttr *CreateImplicit(ASTContext &Ctx, Expr * Min, Expr * Max, SourceRange Range = {}, Spelling S = GNU_amdgpu_flat_work_group_size);
  static AMDGPUFlatWorkGroupSizeAttr *Create(ASTContext &Ctx, Expr * Min, Expr * Max, SourceRange Range = {}, Spelling S = GNU_amdgpu_flat_work_group_size);

  // Constructors
  AMDGPUFlatWorkGroupSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Min
              , Expr * Max
             );

  AMDGPUFlatWorkGroupSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getMin() const {
    return min;
  }

  Expr * getMax() const {
    return max;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUFlatWorkGroupSize; }
};

class CLANG_ABI AMDGPUKernelCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_amdgpu_kernel = 0,
    CXX11_clang_amdgpu_kernel = 1,
    C23_clang_amdgpu_kernel = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUKernelCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AMDGPUKernelCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AMDGPUKernelCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_amdgpu_kernel);
  static AMDGPUKernelCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_amdgpu_kernel);

  // Constructors
  AMDGPUKernelCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AMDGPUKernelCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUKernelCall; }
};

class CLANG_ABI AMDGPUMaxNumWorkGroupsAttr : public InheritableAttr {
Expr * maxNumWorkGroupsX;

Expr * maxNumWorkGroupsY;

Expr * maxNumWorkGroupsZ;

public:
  enum Spelling {
    GNU_amdgpu_max_num_work_groups = 0,
    CXX11_clang_amdgpu_max_num_work_groups = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUMaxNumWorkGroupsAttr *CreateImplicit(ASTContext &Ctx, Expr * MaxNumWorkGroupsX, Expr * MaxNumWorkGroupsY, Expr * MaxNumWorkGroupsZ, const AttributeCommonInfo &CommonInfo);
  static AMDGPUMaxNumWorkGroupsAttr *Create(ASTContext &Ctx, Expr * MaxNumWorkGroupsX, Expr * MaxNumWorkGroupsY, Expr * MaxNumWorkGroupsZ, const AttributeCommonInfo &CommonInfo);
  static AMDGPUMaxNumWorkGroupsAttr *CreateImplicit(ASTContext &Ctx, Expr * MaxNumWorkGroupsX, Expr * MaxNumWorkGroupsY, Expr * MaxNumWorkGroupsZ, SourceRange Range = {}, Spelling S = GNU_amdgpu_max_num_work_groups);
  static AMDGPUMaxNumWorkGroupsAttr *Create(ASTContext &Ctx, Expr * MaxNumWorkGroupsX, Expr * MaxNumWorkGroupsY, Expr * MaxNumWorkGroupsZ, SourceRange Range = {}, Spelling S = GNU_amdgpu_max_num_work_groups);

  // Constructors
  AMDGPUMaxNumWorkGroupsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * MaxNumWorkGroupsX
              , Expr * MaxNumWorkGroupsY
              , Expr * MaxNumWorkGroupsZ
             );
  AMDGPUMaxNumWorkGroupsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * MaxNumWorkGroupsX
             );

  AMDGPUMaxNumWorkGroupsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getMaxNumWorkGroupsX() const {
    return maxNumWorkGroupsX;
  }

  Expr * getMaxNumWorkGroupsY() const {
    return maxNumWorkGroupsY;
  }

  Expr * getMaxNumWorkGroupsZ() const {
    return maxNumWorkGroupsZ;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUMaxNumWorkGroups; }
};

class CLANG_ABI AMDGPUNumSGPRAttr : public InheritableAttr {
unsigned numSGPR;

public:
  enum Spelling {
    GNU_amdgpu_num_sgpr = 0,
    CXX11_clang_amdgpu_num_sgpr = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUNumSGPRAttr *CreateImplicit(ASTContext &Ctx, unsigned NumSGPR, const AttributeCommonInfo &CommonInfo);
  static AMDGPUNumSGPRAttr *Create(ASTContext &Ctx, unsigned NumSGPR, const AttributeCommonInfo &CommonInfo);
  static AMDGPUNumSGPRAttr *CreateImplicit(ASTContext &Ctx, unsigned NumSGPR, SourceRange Range = {}, Spelling S = GNU_amdgpu_num_sgpr);
  static AMDGPUNumSGPRAttr *Create(ASTContext &Ctx, unsigned NumSGPR, SourceRange Range = {}, Spelling S = GNU_amdgpu_num_sgpr);

  // Constructors
  AMDGPUNumSGPRAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned NumSGPR
             );

  AMDGPUNumSGPRAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getNumSGPR() const {
    return numSGPR;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUNumSGPR; }
};

class CLANG_ABI AMDGPUNumVGPRAttr : public InheritableAttr {
unsigned numVGPR;

public:
  enum Spelling {
    GNU_amdgpu_num_vgpr = 0,
    CXX11_clang_amdgpu_num_vgpr = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUNumVGPRAttr *CreateImplicit(ASTContext &Ctx, unsigned NumVGPR, const AttributeCommonInfo &CommonInfo);
  static AMDGPUNumVGPRAttr *Create(ASTContext &Ctx, unsigned NumVGPR, const AttributeCommonInfo &CommonInfo);
  static AMDGPUNumVGPRAttr *CreateImplicit(ASTContext &Ctx, unsigned NumVGPR, SourceRange Range = {}, Spelling S = GNU_amdgpu_num_vgpr);
  static AMDGPUNumVGPRAttr *Create(ASTContext &Ctx, unsigned NumVGPR, SourceRange Range = {}, Spelling S = GNU_amdgpu_num_vgpr);

  // Constructors
  AMDGPUNumVGPRAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned NumVGPR
             );

  AMDGPUNumVGPRAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getNumVGPR() const {
    return numVGPR;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUNumVGPR; }
};

class CLANG_ABI AMDGPUWavesPerEUAttr : public InheritableAttr {
Expr * min;

Expr * max;

public:
  enum Spelling {
    GNU_amdgpu_waves_per_eu = 0,
    CXX11_clang_amdgpu_waves_per_eu = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AMDGPUWavesPerEUAttr *CreateImplicit(ASTContext &Ctx, Expr * Min, Expr * Max, const AttributeCommonInfo &CommonInfo);
  static AMDGPUWavesPerEUAttr *Create(ASTContext &Ctx, Expr * Min, Expr * Max, const AttributeCommonInfo &CommonInfo);
  static AMDGPUWavesPerEUAttr *CreateImplicit(ASTContext &Ctx, Expr * Min, Expr * Max, SourceRange Range = {}, Spelling S = GNU_amdgpu_waves_per_eu);
  static AMDGPUWavesPerEUAttr *Create(ASTContext &Ctx, Expr * Min, Expr * Max, SourceRange Range = {}, Spelling S = GNU_amdgpu_waves_per_eu);

  // Constructors
  AMDGPUWavesPerEUAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Min
              , Expr * Max
             );
  AMDGPUWavesPerEUAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Min
             );

  AMDGPUWavesPerEUAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getMin() const {
    return min;
  }

  Expr * getMax() const {
    return max;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AMDGPUWavesPerEU; }
};

class CLANG_ABI ARMInterruptAttr : public InheritableAttr {
public:
  enum InterruptType {
    IRQ,
    FIQ,
    SWI,
    ABORT,
    UNDEF,
    Generic
  };
private:
  ARMInterruptAttr::InterruptType interrupt;

public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ARMInterruptAttr *CreateImplicit(ASTContext &Ctx, ARMInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static ARMInterruptAttr *Create(ASTContext &Ctx, ARMInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static ARMInterruptAttr *CreateImplicit(ASTContext &Ctx, ARMInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static ARMInterruptAttr *Create(ASTContext &Ctx, ARMInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  ARMInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ARMInterruptAttr::InterruptType Interrupt
             );
  ARMInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ARMInterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ARMInterruptAttr::InterruptType getInterrupt() const {
    return interrupt;
  }

  static bool ConvertStrToInterruptType(StringRef Val, ARMInterruptAttr::InterruptType &Out);
  static const char *ConvertInterruptTypeToStr(ARMInterruptAttr::InterruptType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ARMInterrupt; }
};

class CLANG_ABI ARMInterruptSaveFPAttr : public InheritableAttr {
public:
  enum InterruptType {
    IRQ,
    FIQ,
    SWI,
    ABORT,
    UNDEF,
    Generic
  };
private:
  ARMInterruptSaveFPAttr::InterruptType interrupt;

public:
  // Factory methods
  static ARMInterruptSaveFPAttr *CreateImplicit(ASTContext &Ctx, ARMInterruptSaveFPAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static ARMInterruptSaveFPAttr *Create(ASTContext &Ctx, ARMInterruptSaveFPAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static ARMInterruptSaveFPAttr *CreateImplicit(ASTContext &Ctx, ARMInterruptSaveFPAttr::InterruptType Interrupt, SourceRange Range = {});
  static ARMInterruptSaveFPAttr *Create(ASTContext &Ctx, ARMInterruptSaveFPAttr::InterruptType Interrupt, SourceRange Range = {});

  // Constructors
  ARMInterruptSaveFPAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ARMInterruptSaveFPAttr::InterruptType Interrupt
             );
  ARMInterruptSaveFPAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ARMInterruptSaveFPAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ARMInterruptSaveFPAttr::InterruptType getInterrupt() const {
    return interrupt;
  }

  static bool ConvertStrToInterruptType(StringRef Val, ARMInterruptSaveFPAttr::InterruptType &Out);
  static const char *ConvertInterruptTypeToStr(ARMInterruptSaveFPAttr::InterruptType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ARMInterruptSaveFP; }
};

class CLANG_ABI ARMSaveFPAttr : public InheritableAttr {
public:
  // Factory methods
  static ARMSaveFPAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ARMSaveFPAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ARMSaveFPAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ARMSaveFPAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ARMSaveFPAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ARMSaveFPAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ARMSaveFP; }
};

class CLANG_ABI AVRInterruptAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AVRInterruptAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AVRInterruptAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AVRInterruptAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static AVRInterruptAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  AVRInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AVRInterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AVRInterrupt; }
};

class CLANG_ABI AVRSignalAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_signal = 0,
    CXX11_gnu_signal = 1,
    C23_gnu_signal = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AVRSignalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AVRSignalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AVRSignalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_signal);
  static AVRSignalAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_signal);

  // Constructors
  AVRSignalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AVRSignalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AVRSignal; }
};

class CLANG_ABI AbiTagAttr : public Attr {
  unsigned tags_Size;
  StringRef *tags_;

public:
  enum Spelling {
    GNU_abi_tag = 0,
    CXX11_gnu_abi_tag = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AbiTagAttr *CreateImplicit(ASTContext &Ctx, StringRef *Tags, unsigned TagsSize, const AttributeCommonInfo &CommonInfo);
  static AbiTagAttr *Create(ASTContext &Ctx, StringRef *Tags, unsigned TagsSize, const AttributeCommonInfo &CommonInfo);
  static AbiTagAttr *CreateImplicit(ASTContext &Ctx, StringRef *Tags, unsigned TagsSize, SourceRange Range = {}, Spelling S = GNU_abi_tag);
  static AbiTagAttr *Create(ASTContext &Ctx, StringRef *Tags, unsigned TagsSize, SourceRange Range = {}, Spelling S = GNU_abi_tag);

  // Constructors
  AbiTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *Tags, unsigned TagsSize
             );
  AbiTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AbiTagAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* tags_iterator;
  tags_iterator tags_begin() const { return tags_; }
  tags_iterator tags_end() const { return tags_ + tags_Size; }
  unsigned tags_size() const { return tags_Size; }
  llvm::iterator_range<tags_iterator> tags() const { return llvm::make_range(tags_begin(), tags_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::AbiTag; }
};

class CLANG_ABI AcquireCapabilityAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  enum Spelling {
    GNU_acquire_capability = 0,
    CXX11_clang_acquire_capability = 1,
    GNU_acquire_shared_capability = 2,
    CXX11_clang_acquire_shared_capability = 3,
    GNU_exclusive_lock_function = 4,
    GNU_shared_lock_function = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AcquireCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquireCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquireCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_acquire_capability);
  static AcquireCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_acquire_capability);

  // Constructors
  AcquireCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  AcquireCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AcquireCapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 5; }
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::AcquireCapability; }
};

class CLANG_ABI AcquireHandleAttr : public InheritableAttr {
unsigned handleTypeLength;
char *handleType;

public:
  enum Spelling {
    GNU_acquire_handle = 0,
    CXX11_clang_acquire_handle = 1,
    C23_clang_acquire_handle = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AcquireHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static AcquireHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static AcquireHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_acquire_handle);
  static AcquireHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_acquire_handle);

  // Constructors
  AcquireHandleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef HandleType
             );

  AcquireHandleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getHandleType() const {
    return llvm::StringRef(handleType, handleTypeLength);
  }
  unsigned getHandleTypeLength() const {
    return handleTypeLength;
  }
  void setHandleType(ASTContext &C, llvm::StringRef S) {
    handleTypeLength = S.size();
    this->handleType = new (C, 1) char [handleTypeLength];
    if (!S.empty())
      std::memcpy(this->handleType, S.data(), handleTypeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AcquireHandle; }
};

class CLANG_ABI AcquiredAfterAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  // Factory methods
  static AcquiredAfterAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquiredAfterAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquiredAfterAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});
  static AcquiredAfterAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});

  // Constructors
  AcquiredAfterAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  AcquiredAfterAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AcquiredAfterAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::AcquiredAfter; }
};

class CLANG_ABI AcquiredBeforeAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  // Factory methods
  static AcquiredBeforeAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquiredBeforeAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AcquiredBeforeAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});
  static AcquiredBeforeAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});

  // Constructors
  AcquiredBeforeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  AcquiredBeforeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AcquiredBeforeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::AcquiredBefore; }
};

class CLANG_ABI AddressSpaceAttr : public TypeAttr {
int addressSpace;

public:
  enum Spelling {
    GNU_address_space = 0,
    CXX11_clang_address_space = 1,
    C23_clang_address_space = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AddressSpaceAttr *CreateImplicit(ASTContext &Ctx, int AddressSpace, const AttributeCommonInfo &CommonInfo);
  static AddressSpaceAttr *Create(ASTContext &Ctx, int AddressSpace, const AttributeCommonInfo &CommonInfo);
  static AddressSpaceAttr *CreateImplicit(ASTContext &Ctx, int AddressSpace, SourceRange Range = {}, Spelling S = GNU_address_space);
  static AddressSpaceAttr *Create(ASTContext &Ctx, int AddressSpace, SourceRange Range = {}, Spelling S = GNU_address_space);

  // Constructors
  AddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int AddressSpace
             );

  AddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getAddressSpace() const {
    return addressSpace;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AddressSpace; }
};

class CLANG_ABI AliasAttr : public Attr {
unsigned aliaseeLength;
char *aliasee;

public:
  enum Spelling {
    GNU_alias = 0,
    CXX11_gnu_alias = 1,
    C23_gnu_alias = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AliasAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Aliasee, const AttributeCommonInfo &CommonInfo);
  static AliasAttr *Create(ASTContext &Ctx, llvm::StringRef Aliasee, const AttributeCommonInfo &CommonInfo);
  static AliasAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Aliasee, SourceRange Range = {}, Spelling S = GNU_alias);
  static AliasAttr *Create(ASTContext &Ctx, llvm::StringRef Aliasee, SourceRange Range = {}, Spelling S = GNU_alias);

  // Constructors
  AliasAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Aliasee
             );

  AliasAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAliasee() const {
    return llvm::StringRef(aliasee, aliaseeLength);
  }
  unsigned getAliaseeLength() const {
    return aliaseeLength;
  }
  void setAliasee(ASTContext &C, llvm::StringRef S) {
    aliaseeLength = S.size();
    this->aliasee = new (C, 1) char [aliaseeLength];
    if (!S.empty())
      std::memcpy(this->aliasee, S.data(), aliaseeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Alias; }
};

class CLANG_ABI AlignMac68kAttr : public InheritableAttr {
public:
  // Factory methods
  static AlignMac68kAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlignMac68kAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlignMac68kAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static AlignMac68kAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  AlignMac68kAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AlignMac68kAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AlignMac68k; }
};

class CLANG_ABI AlignNaturalAttr : public InheritableAttr {
public:
  // Factory methods
  static AlignNaturalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlignNaturalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlignNaturalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static AlignNaturalAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  AlignNaturalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AlignNaturalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AlignNatural; }
};

class CLANG_ABI AlignValueAttr : public Attr {
Expr * alignment;

public:
  // Factory methods
  static AlignValueAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static AlignValueAttr *Create(ASTContext &Ctx, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static AlignValueAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, SourceRange Range = {});
  static AlignValueAttr *Create(ASTContext &Ctx, Expr * Alignment, SourceRange Range = {});

  // Constructors
  AlignValueAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Alignment
             );

  AlignValueAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getAlignment() const {
    return alignment;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AlignValue; }
};

class CLANG_ABI AlignedAttr : public InheritableAttr {
bool isalignmentExpr;
union {
Expr *alignmentExpr;
TypeSourceInfo *alignmentType;
};
std::optional<unsigned> alignmentCache;


public:
  enum Spelling {
    GNU_aligned = 0,
    CXX11_gnu_aligned = 1,
    C23_gnu_aligned = 2,
    Declspec_align = 3,
    Keyword_alignas = 4,
    Keyword_Alignas = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AlignedAttr *CreateImplicit(ASTContext &Ctx, bool IsAlignmentExpr, void *Alignment, const AttributeCommonInfo &CommonInfo);
  static AlignedAttr *Create(ASTContext &Ctx, bool IsAlignmentExpr, void *Alignment, const AttributeCommonInfo &CommonInfo);
  static AlignedAttr *CreateImplicit(ASTContext &Ctx, bool IsAlignmentExpr, void *Alignment, SourceRange Range = {}, Spelling S = GNU_aligned);
  static AlignedAttr *Create(ASTContext &Ctx, bool IsAlignmentExpr, void *Alignment, SourceRange Range = {}, Spelling S = GNU_aligned);

  // Constructors
  AlignedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , bool IsAlignmentExpr, void *Alignment
             );
  AlignedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AlignedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isGNU() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1 ||
    getAttributeSpellingListIndex() == 2; }
  bool isC11() const { return getAttributeSpellingListIndex() == 5; }
  bool isAlignas() const { return getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }
  bool isDeclspec() const { return getAttributeSpellingListIndex() == 3; }
  bool isAlignmentDependent() const;
  bool isAlignmentErrorDependent() const;
  unsigned getAlignment(ASTContext &Ctx) const;
  bool isAlignmentExpr() const {
    return isalignmentExpr;
  }
  Expr *getAlignmentExpr() const {
    assert(isalignmentExpr);
    return alignmentExpr;
  }
  TypeSourceInfo *getAlignmentType() const {
    assert(!isalignmentExpr);
    return alignmentType;
  }  std::optional<unsigned> getCachedAlignmentValue() const {
    return alignmentCache;
  }  void setCachedAlignmentValue(unsigned AlignVal) {
    alignmentCache = AlignVal;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Aligned; }
};

class CLANG_ABI AllocAlignAttr : public InheritableAttr {
ParamIdx paramIndex;

public:
  enum Spelling {
    GNU_alloc_align = 0,
    CXX11_gnu_alloc_align = 1,
    C23_gnu_alloc_align = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AllocAlignAttr *CreateImplicit(ASTContext &Ctx, ParamIdx ParamIndex, const AttributeCommonInfo &CommonInfo);
  static AllocAlignAttr *Create(ASTContext &Ctx, ParamIdx ParamIndex, const AttributeCommonInfo &CommonInfo);
  static AllocAlignAttr *CreateImplicit(ASTContext &Ctx, ParamIdx ParamIndex, SourceRange Range = {}, Spelling S = GNU_alloc_align);
  static AllocAlignAttr *Create(ASTContext &Ctx, ParamIdx ParamIndex, SourceRange Range = {}, Spelling S = GNU_alloc_align);

  // Constructors
  AllocAlignAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamIdx ParamIndex
             );

  AllocAlignAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ParamIdx getParamIndex() const {
    return paramIndex;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AllocAlign; }
};

class CLANG_ABI AllocSizeAttr : public InheritableAttr {
ParamIdx elemSizeParam;

ParamIdx numElemsParam;

public:
  enum Spelling {
    GNU_alloc_size = 0,
    CXX11_gnu_alloc_size = 1,
    C23_gnu_alloc_size = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AllocSizeAttr *CreateImplicit(ASTContext &Ctx, ParamIdx ElemSizeParam, ParamIdx NumElemsParam, const AttributeCommonInfo &CommonInfo);
  static AllocSizeAttr *Create(ASTContext &Ctx, ParamIdx ElemSizeParam, ParamIdx NumElemsParam, const AttributeCommonInfo &CommonInfo);
  static AllocSizeAttr *CreateImplicit(ASTContext &Ctx, ParamIdx ElemSizeParam, ParamIdx NumElemsParam, SourceRange Range = {}, Spelling S = GNU_alloc_size);
  static AllocSizeAttr *Create(ASTContext &Ctx, ParamIdx ElemSizeParam, ParamIdx NumElemsParam, SourceRange Range = {}, Spelling S = GNU_alloc_size);

  // Constructors
  AllocSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamIdx ElemSizeParam
              , ParamIdx NumElemsParam
             );
  AllocSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamIdx ElemSizeParam
             );

  AllocSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ParamIdx getElemSizeParam() const {
    return elemSizeParam;
  }

  ParamIdx getNumElemsParam() const {
    return numElemsParam;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AllocSize; }
};

class CLANG_ABI AllocatingAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_allocating = 0,
    CXX11_clang_allocating = 1,
    C23_clang_allocating = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AllocatingAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AllocatingAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AllocatingAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_allocating);
  static AllocatingAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_allocating);

  // Constructors
  AllocatingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AllocatingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Allocating; }
};

class CLANG_ABI AlwaysDestroyAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_always_destroy = 0,
    CXX11_clang_always_destroy = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AlwaysDestroyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlwaysDestroyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlwaysDestroyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_always_destroy);
  static AlwaysDestroyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_always_destroy);

  // Constructors
  AlwaysDestroyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AlwaysDestroyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AlwaysDestroy; }
};

class CLANG_ABI AlwaysInlineAttr : public DeclOrStmtAttr {
public:
  enum Spelling {
    GNU_always_inline = 0,
    CXX11_gnu_always_inline = 1,
    C23_gnu_always_inline = 2,
    CXX11_clang_always_inline = 3,
    C23_clang_always_inline = 4,
    Keyword_forceinline = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AlwaysInlineAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlwaysInlineAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AlwaysInlineAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_always_inline);
  static AlwaysInlineAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_always_inline);

  // Constructors
  AlwaysInlineAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AlwaysInlineAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isClangAlwaysInline() const { return getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 4; }


  static bool classof(const Attr *A) { return A->getKind() == attr::AlwaysInline; }
};

class CLANG_ABI AnalyzerNoReturnAttr : public InheritableAttr {
public:
  // Factory methods
  static AnalyzerNoReturnAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnalyzerNoReturnAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnalyzerNoReturnAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static AnalyzerNoReturnAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  AnalyzerNoReturnAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnalyzerNoReturnAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AnalyzerNoReturn; }
};

class CLANG_ABI AnnotateAttr : public InheritableParamOrStmtAttr {
unsigned annotationLength;
char *annotation;

  unsigned args_Size;
  Expr * *args_;

  unsigned delayedArgs_Size;
  Expr * *delayedArgs_;

public:
  enum Spelling {
    GNU_annotate = 0,
    CXX11_clang_annotate = 1,
    C23_clang_annotate = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AnnotateAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateAttr *Create(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_annotate);
  static AnnotateAttr *Create(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_annotate);
  static AnnotateAttr *CreateImplicitWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateAttr *CreateWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateAttr *CreateImplicitWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, SourceRange Range = {}, Spelling S = GNU_annotate);
  static AnnotateAttr *CreateWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, SourceRange Range = {}, Spelling S = GNU_annotate);

  // Constructors
  AnnotateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Annotation
              , Expr * *Args, unsigned ArgsSize
             );
  AnnotateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Annotation
             );
  AnnotateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnnotateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAnnotation() const {
    return llvm::StringRef(annotation, annotationLength);
  }
  unsigned getAnnotationLength() const {
    return annotationLength;
  }
  void setAnnotation(ASTContext &C, llvm::StringRef S) {
    annotationLength = S.size();
    this->annotation = new (C, 1) char [annotationLength];
    if (!S.empty())
      std::memcpy(this->annotation, S.data(), annotationLength);
  }

  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }


  typedef Expr ** delayedArgs_iterator;
  delayedArgs_iterator delayedArgs_begin() const { return delayedArgs_; }
  delayedArgs_iterator delayedArgs_end() const { return delayedArgs_ + delayedArgs_Size; }
  unsigned delayedArgs_size() const { return delayedArgs_Size; }
  llvm::iterator_range<delayedArgs_iterator> delayedArgs() const { return llvm::make_range(delayedArgs_begin(), delayedArgs_end()); }
  void setDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize) {
    delayedArgs_Size = DelayedArgsSize;
    delayedArgs_ = new (Ctx, 16) Expr *[delayedArgs_Size];
    std::copy(DelayedArgs, DelayedArgs + delayedArgs_Size, delayedArgs_);
  }

  static AnnotateAttr *Create(ASTContext &Ctx, llvm::StringRef Annotation, \
              const AttributeCommonInfo &CommonInfo) {
    return AnnotateAttr::Create(Ctx, Annotation, nullptr, 0, CommonInfo);
  }
  static AnnotateAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Annotation, \
              const AttributeCommonInfo &CommonInfo) {
    return AnnotateAttr::CreateImplicit(Ctx, Annotation, nullptr, 0, CommonInfo);
  }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::Annotate; }
};

class CLANG_ABI AnnotateTypeAttr : public TypeAttr {
unsigned annotationLength;
char *annotation;

  unsigned args_Size;
  Expr * *args_;

  unsigned delayedArgs_Size;
  Expr * *delayedArgs_;

public:
  enum Spelling {
    CXX11_clang_annotate_type = 0,
    C23_clang_annotate_type = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AnnotateTypeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateTypeAttr *Create(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateTypeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = CXX11_clang_annotate_type);
  static AnnotateTypeAttr *Create(ASTContext &Ctx, llvm::StringRef Annotation, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = CXX11_clang_annotate_type);
  static AnnotateTypeAttr *CreateImplicitWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateTypeAttr *CreateWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, const AttributeCommonInfo &CommonInfo);
  static AnnotateTypeAttr *CreateImplicitWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, SourceRange Range = {}, Spelling S = CXX11_clang_annotate_type);
  static AnnotateTypeAttr *CreateWithDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize, SourceRange Range = {}, Spelling S = CXX11_clang_annotate_type);

  // Constructors
  AnnotateTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Annotation
              , Expr * *Args, unsigned ArgsSize
             );
  AnnotateTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Annotation
             );
  AnnotateTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnnotateTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAnnotation() const {
    return llvm::StringRef(annotation, annotationLength);
  }
  unsigned getAnnotationLength() const {
    return annotationLength;
  }
  void setAnnotation(ASTContext &C, llvm::StringRef S) {
    annotationLength = S.size();
    this->annotation = new (C, 1) char [annotationLength];
    if (!S.empty())
      std::memcpy(this->annotation, S.data(), annotationLength);
  }

  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }


  typedef Expr ** delayedArgs_iterator;
  delayedArgs_iterator delayedArgs_begin() const { return delayedArgs_; }
  delayedArgs_iterator delayedArgs_end() const { return delayedArgs_ + delayedArgs_Size; }
  unsigned delayedArgs_size() const { return delayedArgs_Size; }
  llvm::iterator_range<delayedArgs_iterator> delayedArgs() const { return llvm::make_range(delayedArgs_begin(), delayedArgs_end()); }
  void setDelayedArgs(ASTContext &Ctx, Expr * *DelayedArgs, unsigned DelayedArgsSize) {
    delayedArgs_Size = DelayedArgsSize;
    delayedArgs_ = new (Ctx, 16) Expr *[delayedArgs_Size];
    std::copy(DelayedArgs, DelayedArgs + delayedArgs_Size, delayedArgs_);
  }


  static bool classof(const Attr *A) { return A->getKind() == attr::AnnotateType; }
};

class CLANG_ABI AnyX86InterruptAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AnyX86InterruptAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86InterruptAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86InterruptAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static AnyX86InterruptAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  AnyX86InterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnyX86InterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AnyX86Interrupt; }
};

class CLANG_ABI AnyX86NoCallerSavedRegistersAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_caller_saved_registers = 0,
    CXX11_gnu_no_caller_saved_registers = 1,
    C23_gnu_no_caller_saved_registers = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AnyX86NoCallerSavedRegistersAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86NoCallerSavedRegistersAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86NoCallerSavedRegistersAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_caller_saved_registers);
  static AnyX86NoCallerSavedRegistersAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_caller_saved_registers);

  // Constructors
  AnyX86NoCallerSavedRegistersAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnyX86NoCallerSavedRegistersAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AnyX86NoCallerSavedRegisters; }
};

class CLANG_ABI AnyX86NoCfCheckAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nocf_check = 0,
    CXX11_gnu_nocf_check = 1,
    C23_gnu_nocf_check = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AnyX86NoCfCheckAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86NoCfCheckAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AnyX86NoCfCheckAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nocf_check);
  static AnyX86NoCfCheckAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nocf_check);

  // Constructors
  AnyX86NoCfCheckAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AnyX86NoCfCheckAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AnyX86NoCfCheck; }
};

class CLANG_ABI ArcWeakrefUnavailableAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_arc_weak_reference_unavailable = 0,
    CXX11_clang_objc_arc_weak_reference_unavailable = 1,
    C23_clang_objc_arc_weak_reference_unavailable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ArcWeakrefUnavailableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArcWeakrefUnavailableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArcWeakrefUnavailableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_arc_weak_reference_unavailable);
  static ArcWeakrefUnavailableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_arc_weak_reference_unavailable);

  // Constructors
  ArcWeakrefUnavailableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArcWeakrefUnavailableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ArcWeakrefUnavailable; }
};

class CLANG_ABI ArgumentWithTypeTagAttr : public InheritableAttr {
IdentifierInfo * argumentKind;

ParamIdx argumentIdx;

ParamIdx typeTagIdx;

bool isPointer;

public:
  enum Spelling {
    GNU_argument_with_type_tag = 0,
    CXX11_clang_argument_with_type_tag = 1,
    C23_clang_argument_with_type_tag = 2,
    GNU_pointer_with_type_tag = 3,
    CXX11_clang_pointer_with_type_tag = 4,
    C23_clang_pointer_with_type_tag = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ArgumentWithTypeTagAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, bool IsPointer, const AttributeCommonInfo &CommonInfo);
  static ArgumentWithTypeTagAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, bool IsPointer, const AttributeCommonInfo &CommonInfo);
  static ArgumentWithTypeTagAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, bool IsPointer, SourceRange Range = {}, Spelling S = GNU_argument_with_type_tag);
  static ArgumentWithTypeTagAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, bool IsPointer, SourceRange Range = {}, Spelling S = GNU_argument_with_type_tag);
  static ArgumentWithTypeTagAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, const AttributeCommonInfo &CommonInfo);
  static ArgumentWithTypeTagAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, const AttributeCommonInfo &CommonInfo);
  static ArgumentWithTypeTagAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, SourceRange Range = {}, Spelling S = GNU_argument_with_type_tag);
  static ArgumentWithTypeTagAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, ParamIdx ArgumentIdx, ParamIdx TypeTagIdx, SourceRange Range = {}, Spelling S = GNU_argument_with_type_tag);

  // Constructors
  ArgumentWithTypeTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * ArgumentKind
              , ParamIdx ArgumentIdx
              , ParamIdx TypeTagIdx
              , bool IsPointer
             );
  ArgumentWithTypeTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * ArgumentKind
              , ParamIdx ArgumentIdx
              , ParamIdx TypeTagIdx
             );

  ArgumentWithTypeTagAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  IdentifierInfo * getArgumentKind() const {
    return argumentKind;
  }

  ParamIdx getArgumentIdx() const {
    return argumentIdx;
  }

  ParamIdx getTypeTagIdx() const {
    return typeTagIdx;
  }

  bool getIsPointer() const {
    return isPointer;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ArgumentWithTypeTag; }
};

class CLANG_ABI ArmAgnosticAttr : public TypeAttr {
  unsigned agnosticArgs_Size;
  StringRef *agnosticArgs_;

public:
  // Factory methods
  static ArmAgnosticAttr *CreateImplicit(ASTContext &Ctx, StringRef *AgnosticArgs, unsigned AgnosticArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmAgnosticAttr *Create(ASTContext &Ctx, StringRef *AgnosticArgs, unsigned AgnosticArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmAgnosticAttr *CreateImplicit(ASTContext &Ctx, StringRef *AgnosticArgs, unsigned AgnosticArgsSize, SourceRange Range = {});
  static ArmAgnosticAttr *Create(ASTContext &Ctx, StringRef *AgnosticArgs, unsigned AgnosticArgsSize, SourceRange Range = {});

  // Constructors
  ArmAgnosticAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *AgnosticArgs, unsigned AgnosticArgsSize
             );
  ArmAgnosticAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmAgnosticAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* agnosticArgs_iterator;
  agnosticArgs_iterator agnosticArgs_begin() const { return agnosticArgs_; }
  agnosticArgs_iterator agnosticArgs_end() const { return agnosticArgs_ + agnosticArgs_Size; }
  unsigned agnosticArgs_size() const { return agnosticArgs_Size; }
  llvm::iterator_range<agnosticArgs_iterator> agnosticArgs() const { return llvm::make_range(agnosticArgs_begin(), agnosticArgs_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ArmAgnostic; }
};

class CLANG_ABI ArmBuiltinAliasAttr : public InheritableAttr {
IdentifierInfo * builtinName;

public:
  enum Spelling {
    GNU_clang_arm_builtin_alias = 0,
    CXX11_clang_clang_arm_builtin_alias = 1,
    C23_clang_clang_arm_builtin_alias = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ArmBuiltinAliasAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BuiltinName, const AttributeCommonInfo &CommonInfo);
  static ArmBuiltinAliasAttr *Create(ASTContext &Ctx, IdentifierInfo * BuiltinName, const AttributeCommonInfo &CommonInfo);
  static ArmBuiltinAliasAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BuiltinName, SourceRange Range = {}, Spelling S = GNU_clang_arm_builtin_alias);
  static ArmBuiltinAliasAttr *Create(ASTContext &Ctx, IdentifierInfo * BuiltinName, SourceRange Range = {}, Spelling S = GNU_clang_arm_builtin_alias);

  // Constructors
  ArmBuiltinAliasAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * BuiltinName
             );

  ArmBuiltinAliasAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getBuiltinName() const {
    return builtinName;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ArmBuiltinAlias; }
};

class CLANG_ABI ArmInAttr : public TypeAttr {
  unsigned inArgs_Size;
  StringRef *inArgs_;

public:
  // Factory methods
  static ArmInAttr *CreateImplicit(ASTContext &Ctx, StringRef *InArgs, unsigned InArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmInAttr *Create(ASTContext &Ctx, StringRef *InArgs, unsigned InArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmInAttr *CreateImplicit(ASTContext &Ctx, StringRef *InArgs, unsigned InArgsSize, SourceRange Range = {});
  static ArmInAttr *Create(ASTContext &Ctx, StringRef *InArgs, unsigned InArgsSize, SourceRange Range = {});

  // Constructors
  ArmInAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *InArgs, unsigned InArgsSize
             );
  ArmInAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmInAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* inArgs_iterator;
  inArgs_iterator inArgs_begin() const { return inArgs_; }
  inArgs_iterator inArgs_end() const { return inArgs_ + inArgs_Size; }
  unsigned inArgs_size() const { return inArgs_Size; }
  llvm::iterator_range<inArgs_iterator> inArgs() const { return llvm::make_range(inArgs_begin(), inArgs_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ArmIn; }
};

class CLANG_ABI ArmInOutAttr : public TypeAttr {
  unsigned inOutArgs_Size;
  StringRef *inOutArgs_;

public:
  // Factory methods
  static ArmInOutAttr *CreateImplicit(ASTContext &Ctx, StringRef *InOutArgs, unsigned InOutArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmInOutAttr *Create(ASTContext &Ctx, StringRef *InOutArgs, unsigned InOutArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmInOutAttr *CreateImplicit(ASTContext &Ctx, StringRef *InOutArgs, unsigned InOutArgsSize, SourceRange Range = {});
  static ArmInOutAttr *Create(ASTContext &Ctx, StringRef *InOutArgs, unsigned InOutArgsSize, SourceRange Range = {});

  // Constructors
  ArmInOutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *InOutArgs, unsigned InOutArgsSize
             );
  ArmInOutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmInOutAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* inOutArgs_iterator;
  inOutArgs_iterator inOutArgs_begin() const { return inOutArgs_; }
  inOutArgs_iterator inOutArgs_end() const { return inOutArgs_ + inOutArgs_Size; }
  unsigned inOutArgs_size() const { return inOutArgs_Size; }
  llvm::iterator_range<inOutArgs_iterator> inOutArgs() const { return llvm::make_range(inOutArgs_begin(), inOutArgs_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ArmInOut; }
};

class CLANG_ABI ArmLocallyStreamingAttr : public InheritableAttr {
public:
  // Factory methods
  static ArmLocallyStreamingAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmLocallyStreamingAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmLocallyStreamingAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ArmLocallyStreamingAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ArmLocallyStreamingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmLocallyStreamingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ArmLocallyStreaming; }
};

class CLANG_ABI ArmMveStrictPolymorphismAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_clang_arm_mve_strict_polymorphism = 0,
    CXX11_clang_clang_arm_mve_strict_polymorphism = 1,
    C23_clang_clang_arm_mve_strict_polymorphism = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ArmMveStrictPolymorphismAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmMveStrictPolymorphismAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmMveStrictPolymorphismAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_clang_arm_mve_strict_polymorphism);
  static ArmMveStrictPolymorphismAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_clang_arm_mve_strict_polymorphism);

  // Constructors
  ArmMveStrictPolymorphismAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmMveStrictPolymorphismAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ArmMveStrictPolymorphism; }
};

class CLANG_ABI ArmNewAttr : public InheritableAttr {
  unsigned newArgs_Size;
  StringRef *newArgs_;

public:
  // Factory methods
  static ArmNewAttr *CreateImplicit(ASTContext &Ctx, StringRef *NewArgs, unsigned NewArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmNewAttr *Create(ASTContext &Ctx, StringRef *NewArgs, unsigned NewArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmNewAttr *CreateImplicit(ASTContext &Ctx, StringRef *NewArgs, unsigned NewArgsSize, SourceRange Range = {});
  static ArmNewAttr *Create(ASTContext &Ctx, StringRef *NewArgs, unsigned NewArgsSize, SourceRange Range = {});

  // Constructors
  ArmNewAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *NewArgs, unsigned NewArgsSize
             );
  ArmNewAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmNewAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* newArgs_iterator;
  newArgs_iterator newArgs_begin() const { return newArgs_; }
  newArgs_iterator newArgs_end() const { return newArgs_ + newArgs_Size; }
  unsigned newArgs_size() const { return newArgs_Size; }
  llvm::iterator_range<newArgs_iterator> newArgs() const { return llvm::make_range(newArgs_begin(), newArgs_end()); }



    bool isNewZA() const {
      return llvm::is_contained(newArgs(), "za");
    }
    bool isNewZT0() const {
      return llvm::is_contained(newArgs(), "zt0");
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::ArmNew; }
};

class CLANG_ABI ArmOutAttr : public TypeAttr {
  unsigned outArgs_Size;
  StringRef *outArgs_;

public:
  // Factory methods
  static ArmOutAttr *CreateImplicit(ASTContext &Ctx, StringRef *OutArgs, unsigned OutArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmOutAttr *Create(ASTContext &Ctx, StringRef *OutArgs, unsigned OutArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmOutAttr *CreateImplicit(ASTContext &Ctx, StringRef *OutArgs, unsigned OutArgsSize, SourceRange Range = {});
  static ArmOutAttr *Create(ASTContext &Ctx, StringRef *OutArgs, unsigned OutArgsSize, SourceRange Range = {});

  // Constructors
  ArmOutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *OutArgs, unsigned OutArgsSize
             );
  ArmOutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmOutAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* outArgs_iterator;
  outArgs_iterator outArgs_begin() const { return outArgs_; }
  outArgs_iterator outArgs_end() const { return outArgs_ + outArgs_Size; }
  unsigned outArgs_size() const { return outArgs_Size; }
  llvm::iterator_range<outArgs_iterator> outArgs() const { return llvm::make_range(outArgs_begin(), outArgs_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ArmOut; }
};

class CLANG_ABI ArmPreservesAttr : public TypeAttr {
  unsigned preserveArgs_Size;
  StringRef *preserveArgs_;

public:
  // Factory methods
  static ArmPreservesAttr *CreateImplicit(ASTContext &Ctx, StringRef *PreserveArgs, unsigned PreserveArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmPreservesAttr *Create(ASTContext &Ctx, StringRef *PreserveArgs, unsigned PreserveArgsSize, const AttributeCommonInfo &CommonInfo);
  static ArmPreservesAttr *CreateImplicit(ASTContext &Ctx, StringRef *PreserveArgs, unsigned PreserveArgsSize, SourceRange Range = {});
  static ArmPreservesAttr *Create(ASTContext &Ctx, StringRef *PreserveArgs, unsigned PreserveArgsSize, SourceRange Range = {});

  // Constructors
  ArmPreservesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *PreserveArgs, unsigned PreserveArgsSize
             );
  ArmPreservesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmPreservesAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* preserveArgs_iterator;
  preserveArgs_iterator preserveArgs_begin() const { return preserveArgs_; }
  preserveArgs_iterator preserveArgs_end() const { return preserveArgs_ + preserveArgs_Size; }
  unsigned preserveArgs_size() const { return preserveArgs_Size; }
  llvm::iterator_range<preserveArgs_iterator> preserveArgs() const { return llvm::make_range(preserveArgs_begin(), preserveArgs_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ArmPreserves; }
};

class CLANG_ABI ArmStreamingAttr : public TypeAttr {
public:
  // Factory methods
  static ArmStreamingAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmStreamingAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmStreamingAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ArmStreamingAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ArmStreamingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmStreamingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ArmStreaming; }
};

class CLANG_ABI ArmStreamingCompatibleAttr : public TypeAttr {
public:
  // Factory methods
  static ArmStreamingCompatibleAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmStreamingCompatibleAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArmStreamingCompatibleAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ArmStreamingCompatibleAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ArmStreamingCompatibleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArmStreamingCompatibleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ArmStreamingCompatible; }
};

class CLANG_ABI ArtificialAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_artificial = 0,
    CXX11_gnu_artificial = 1,
    C23_gnu_artificial = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ArtificialAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArtificialAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ArtificialAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_artificial);
  static ArtificialAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_artificial);

  // Constructors
  ArtificialAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ArtificialAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Artificial; }
};

class CLANG_ABI AsmLabelAttr : public InheritableAttr {
unsigned labelLength;
char *label;

bool isLiteralLabel;

public:
  enum Spelling {
    Keyword_asm = 0,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AsmLabelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Label, bool IsLiteralLabel, const AttributeCommonInfo &CommonInfo);
  static AsmLabelAttr *Create(ASTContext &Ctx, llvm::StringRef Label, bool IsLiteralLabel, const AttributeCommonInfo &CommonInfo);
  static AsmLabelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Label, bool IsLiteralLabel, SourceRange Range = {}, Spelling S = Keyword_asm);
  static AsmLabelAttr *Create(ASTContext &Ctx, llvm::StringRef Label, bool IsLiteralLabel, SourceRange Range = {}, Spelling S = Keyword_asm);
  static AsmLabelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Label, const AttributeCommonInfo &CommonInfo);
  static AsmLabelAttr *Create(ASTContext &Ctx, llvm::StringRef Label, const AttributeCommonInfo &CommonInfo);
  static AsmLabelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Label, SourceRange Range = {}, Spelling S = Keyword_asm);
  static AsmLabelAttr *Create(ASTContext &Ctx, llvm::StringRef Label, SourceRange Range = {}, Spelling S = Keyword_asm);

  // Constructors
  AsmLabelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Label
              , bool IsLiteralLabel
             );
  AsmLabelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Label
             );

  AsmLabelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getLabel() const {
    return llvm::StringRef(label, labelLength);
  }
  unsigned getLabelLength() const {
    return labelLength;
  }
  void setLabel(ASTContext &C, llvm::StringRef S) {
    labelLength = S.size();
    this->label = new (C, 1) char [labelLength];
    if (!S.empty())
      std::memcpy(this->label, S.data(), labelLength);
  }

  bool getIsLiteralLabel() const {
    return isLiteralLabel;
  }


bool isEquivalent(AsmLabelAttr *Other) const {
  return getLabel() == Other->getLabel() && getIsLiteralLabel() == Other->getIsLiteralLabel();
}


  static bool classof(const Attr *A) { return A->getKind() == attr::AsmLabel; }
};

class CLANG_ABI AssertCapabilityAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  enum Spelling {
    GNU_assert_capability = 0,
    CXX11_clang_assert_capability = 1,
    GNU_assert_shared_capability = 2,
    CXX11_clang_assert_shared_capability = 3,
    GNU_assert_exclusive_lock = 4,
    GNU_assert_shared_lock = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AssertCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AssertCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static AssertCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_assert_capability);
  static AssertCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_assert_capability);

  // Constructors
  AssertCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  AssertCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AssertCapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 5; }
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::AssertCapability; }
};

class CLANG_ABI AssumeAlignedAttr : public InheritableAttr {
Expr * alignment;

Expr * offset;

public:
  enum Spelling {
    GNU_assume_aligned = 0,
    CXX11_gnu_assume_aligned = 1,
    C23_gnu_assume_aligned = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AssumeAlignedAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, Expr * Offset, const AttributeCommonInfo &CommonInfo);
  static AssumeAlignedAttr *Create(ASTContext &Ctx, Expr * Alignment, Expr * Offset, const AttributeCommonInfo &CommonInfo);
  static AssumeAlignedAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, Expr * Offset, SourceRange Range = {}, Spelling S = GNU_assume_aligned);
  static AssumeAlignedAttr *Create(ASTContext &Ctx, Expr * Alignment, Expr * Offset, SourceRange Range = {}, Spelling S = GNU_assume_aligned);

  // Constructors
  AssumeAlignedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Alignment
              , Expr * Offset
             );
  AssumeAlignedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Alignment
             );

  AssumeAlignedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getAlignment() const {
    return alignment;
  }

  Expr * getOffset() const {
    return offset;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::AssumeAligned; }
};

class CLANG_ABI AtomicAttr : public StmtAttr {
public:
  enum ConsumedOption {
    remote_memory,
    no_remote_memory,
    fine_grained_memory,
    no_fine_grained_memory,
    ignore_denormal_mode,
    no_ignore_denormal_mode
  };
private:
  unsigned atomicOptions_Size;
  ConsumedOption *atomicOptions_;

public:
  enum Spelling {
    GNU_atomic = 0,
    CXX11_clang_atomic = 1,
    C23_clang_atomic = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AtomicAttr *CreateImplicit(ASTContext &Ctx, ConsumedOption *AtomicOptions, unsigned AtomicOptionsSize, const AttributeCommonInfo &CommonInfo);
  static AtomicAttr *Create(ASTContext &Ctx, ConsumedOption *AtomicOptions, unsigned AtomicOptionsSize, const AttributeCommonInfo &CommonInfo);
  static AtomicAttr *CreateImplicit(ASTContext &Ctx, ConsumedOption *AtomicOptions, unsigned AtomicOptionsSize, SourceRange Range = {}, Spelling S = GNU_atomic);
  static AtomicAttr *Create(ASTContext &Ctx, ConsumedOption *AtomicOptions, unsigned AtomicOptionsSize, SourceRange Range = {}, Spelling S = GNU_atomic);

  // Constructors
  AtomicAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ConsumedOption *AtomicOptions, unsigned AtomicOptionsSize
             );
  AtomicAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AtomicAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef ConsumedOption* atomicOptions_iterator;
  atomicOptions_iterator atomicOptions_begin() const { return atomicOptions_; }
  atomicOptions_iterator atomicOptions_end() const { return atomicOptions_ + atomicOptions_Size; }
  unsigned atomicOptions_size() const { return atomicOptions_Size; }
  llvm::iterator_range<atomicOptions_iterator> atomicOptions() const { return llvm::make_range(atomicOptions_begin(), atomicOptions_end()); }


  static bool ConvertStrToConsumedOption(StringRef Val, AtomicAttr::ConsumedOption &Out);
  static const char *ConvertConsumedOptionToStr(AtomicAttr::ConsumedOption Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::Atomic; }
};

class CLANG_ABI AvailabilityAttr : public InheritableAttr {
IdentifierInfo * platform;

VersionTuple introduced;


VersionTuple deprecated;


VersionTuple obsoleted;


bool unavailable;

unsigned messageLength;
char *message;

bool strict;

unsigned replacementLength;
char *replacement;

int priority;

IdentifierInfo * environment;

public:
  enum Spelling {
    GNU_availability = 0,
    CXX11_clang_availability = 1,
    C23_clang_availability = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AvailabilityAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Platform, VersionTuple Introduced, VersionTuple Deprecated, VersionTuple Obsoleted, bool Unavailable, llvm::StringRef Message, bool Strict, llvm::StringRef Replacement, int Priority, IdentifierInfo * Environment, const AttributeCommonInfo &CommonInfo);
  static AvailabilityAttr *Create(ASTContext &Ctx, IdentifierInfo * Platform, VersionTuple Introduced, VersionTuple Deprecated, VersionTuple Obsoleted, bool Unavailable, llvm::StringRef Message, bool Strict, llvm::StringRef Replacement, int Priority, IdentifierInfo * Environment, const AttributeCommonInfo &CommonInfo);
  static AvailabilityAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Platform, VersionTuple Introduced, VersionTuple Deprecated, VersionTuple Obsoleted, bool Unavailable, llvm::StringRef Message, bool Strict, llvm::StringRef Replacement, int Priority, IdentifierInfo * Environment, SourceRange Range = {}, Spelling S = GNU_availability);
  static AvailabilityAttr *Create(ASTContext &Ctx, IdentifierInfo * Platform, VersionTuple Introduced, VersionTuple Deprecated, VersionTuple Obsoleted, bool Unavailable, llvm::StringRef Message, bool Strict, llvm::StringRef Replacement, int Priority, IdentifierInfo * Environment, SourceRange Range = {}, Spelling S = GNU_availability);

  // Constructors
  AvailabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Platform
              , VersionTuple Introduced
              , VersionTuple Deprecated
              , VersionTuple Obsoleted
              , bool Unavailable
              , llvm::StringRef Message
              , bool Strict
              , llvm::StringRef Replacement
              , int Priority
              , IdentifierInfo * Environment
             );

  AvailabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getPlatform() const {
    return platform;
  }

  VersionTuple getIntroduced() const {
    return introduced;
  }
  void setIntroduced(ASTContext &C, VersionTuple V) {
    introduced = V;
  }

  VersionTuple getDeprecated() const {
    return deprecated;
  }
  void setDeprecated(ASTContext &C, VersionTuple V) {
    deprecated = V;
  }

  VersionTuple getObsoleted() const {
    return obsoleted;
  }
  void setObsoleted(ASTContext &C, VersionTuple V) {
    obsoleted = V;
  }

  bool getUnavailable() const {
    return unavailable;
  }

  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }

  bool getStrict() const {
    return strict;
  }

  llvm::StringRef getReplacement() const {
    return llvm::StringRef(replacement, replacementLength);
  }
  unsigned getReplacementLength() const {
    return replacementLength;
  }
  void setReplacement(ASTContext &C, llvm::StringRef S) {
    replacementLength = S.size();
    this->replacement = new (C, 1) char [replacementLength];
    if (!S.empty())
      std::memcpy(this->replacement, S.data(), replacementLength);
  }

  int getPriority() const {
    return priority;
  }

  IdentifierInfo * getEnvironment() const {
    return environment;
  }

static llvm::StringRef getPrettyPlatformName(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("android", "Android")
             .Case("fuchsia", "Fuchsia")
             .Case("ios", "iOS")
             .Case("macos", "macOS")
             .Case("tvos", "tvOS")
             .Case("watchos", "watchOS")
             .Case("driverkit", "DriverKit")
             .Case("ios_app_extension", "iOS (App Extension)")
             .Case("macos_app_extension", "macOS (App Extension)")
             .Case("tvos_app_extension", "tvOS (App Extension)")
             .Case("watchos_app_extension", "watchOS (App Extension)")
             .Case("maccatalyst", "macCatalyst")
             .Case("maccatalyst_app_extension", "macCatalyst (App Extension)")
             .Case("xros", "visionOS")
             .Case("xros_app_extension", "visionOS (App Extension)")
             .Case("swift", "Swift")
             .Case("shadermodel", "Shader Model")
             .Case("ohos", "OpenHarmony OS")
             .Default(llvm::StringRef());
}
static llvm::StringRef getPlatformNameSourceSpelling(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("ios", "iOS")
             .Case("macos", "macOS")
             .Case("tvos", "tvOS")
             .Case("watchos", "watchOS")
             .Case("ios_app_extension", "iOSApplicationExtension")
             .Case("macos_app_extension", "macOSApplicationExtension")
             .Case("tvos_app_extension", "tvOSApplicationExtension")
             .Case("watchos_app_extension", "watchOSApplicationExtension")
             .Case("maccatalyst", "macCatalyst")
             .Case("maccatalyst_app_extension", "macCatalystApplicationExtension")
             .Case("xros", "visionOS")
             .Case("xros_app_extension", "visionOSApplicationExtension")
             .Case("zos", "z/OS")
             .Case("shadermodel", "ShaderModel")
             .Default(Platform);
}
static llvm::StringRef canonicalizePlatformName(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("iOS", "ios")
             .Case("macOS", "macos")
             .Case("tvOS", "tvos")
             .Case("watchOS", "watchos")
             .Case("iOSApplicationExtension", "ios_app_extension")
             .Case("macOSApplicationExtension", "macos_app_extension")
             .Case("tvOSApplicationExtension", "tvos_app_extension")
             .Case("watchOSApplicationExtension", "watchos_app_extension")
             .Case("macCatalyst", "maccatalyst")
             .Case("macCatalystApplicationExtension", "maccatalyst_app_extension")
             .Case("visionOS", "xros")
             .Case("visionOSApplicationExtension", "xros_app_extension")
             .Case("visionos", "xros")
             .Case("visionos_app_extension", "xros_app_extension")
             .Case("ShaderModel", "shadermodel")
             .Default(Platform);
}
static std::vector<llvm::StringRef> equivalentPlatformNames(llvm::StringRef Platform) {
    return llvm::StringSwitch<std::vector<llvm::StringRef>>(Platform)
             .Case("ios", {"ios", "iOS"})
             .Case("iOS", {"ios", "iOS"})
             .Case("macos", {"macos", "macOS"})
             .Case("macOS", {"macos", "macOS"})
             .Case("tvos", {"tvos", "tvOS"})
             .Case("tvOS", {"tvos", "tvOS"})
             .Case("watchos", {"watchos", "watchOS"})
             .Case("watchOS", {"watchos", "watchOS"})
             .Case("ios_app_extension", {"iOSApplicationExtension", "ios_app_extension"})
             .Case("iOSApplicationExtension", {"iOSApplicationExtension", "ios_app_extension"})
             .Case("macos_app_extension", {"macOSApplicationExtension", "macos_app_extension"})
             .Case("macOSApplicationExtension", {"macOSApplicationExtension", "macos_app_extension"})
             .Case("tvos_app_extension", {"tvOSApplicationExtension", "tvos_app_extension"})
             .Case("tvOSApplicationExtension", {"tvOSApplicationExtension", "tvos_app_extension"})
             .Case("watchos_app_extension", {"watchOSApplicationExtension", "watchos_app_extension"})
             .Case("watchOSApplicationExtension", {"watchOSApplicationExtension", "watchos_app_extension"})
             .Case("maccatalyst", {"macCatalyst", "maccatalyst"})
             .Case("macCatalyst", {"macCatalyst", "maccatalyst"})
             .Case("maccatalyst_app_extension", {"macCatalystApplicationExtension", "maccatalyst_app_extension"})
             .Case("macCatalystApplicationExtension", {"macCatalystApplicationExtension", "maccatalyst_app_extension"})
             .Case("xros", {"visionos", "visionOS", "xros"})
             .Case("visionOS", {"visionos", "visionOS", "xros"})
             .Case("visionos", {"visionos", "visionOS", "xros"})
             .Case("xros_app_extension", {"visionOSApplicationExtension", "visionos_app_extension", "xros_app_extension"})
             .Case("visionOSApplicationExtension", {"visionOSApplicationExtension", "visionos_app_extension", "xros_app_extension"})
             .Case("visionos_app_extension", {"visionOSApplicationExtension", "visionos_app_extension", "xros_app_extension"})
             .Default({Platform});
}
static llvm::Triple::EnvironmentType getEnvironmentType(llvm::StringRef Environment) {
    return llvm::StringSwitch<llvm::Triple::EnvironmentType>(Environment)
             .Case("pixel", llvm::Triple::Pixel)
             .Case("vertex", llvm::Triple::Vertex)
             .Case("geometry", llvm::Triple::Geometry)
             .Case("hull", llvm::Triple::Hull)
             .Case("domain", llvm::Triple::Domain)
             .Case("compute", llvm::Triple::Compute)
             .Case("raygeneration", llvm::Triple::RayGeneration)
             .Case("intersection", llvm::Triple::Intersection)
             .Case("anyhit", llvm::Triple::AnyHit)
             .Case("closesthit", llvm::Triple::ClosestHit)
             .Case("miss", llvm::Triple::Miss)
             .Case("callable", llvm::Triple::Callable)
             .Case("mesh", llvm::Triple::Mesh)
             .Case("amplification", llvm::Triple::Amplification)
             .Case("library", llvm::Triple::Library)
             .Default(llvm::Triple::UnknownEnvironment);
}


  static bool classof(const Attr *A) { return A->getKind() == attr::Availability; }
};

class CLANG_ABI AvailableOnlyInDefaultEvalMethodAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_available_only_in_default_eval_method = 0,
    CXX11_clang_available_only_in_default_eval_method = 1,
    C23_clang_available_only_in_default_eval_method = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static AvailableOnlyInDefaultEvalMethodAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AvailableOnlyInDefaultEvalMethodAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static AvailableOnlyInDefaultEvalMethodAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_available_only_in_default_eval_method);
  static AvailableOnlyInDefaultEvalMethodAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_available_only_in_default_eval_method);

  // Constructors
  AvailableOnlyInDefaultEvalMethodAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  AvailableOnlyInDefaultEvalMethodAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::AvailableOnlyInDefaultEvalMethod; }
};

class CLANG_ABI BPFFastCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_bpf_fastcall = 0,
    CXX11_clang_bpf_fastcall = 1,
    C23_clang_bpf_fastcall = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BPFFastCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFFastCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFFastCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_bpf_fastcall);
  static BPFFastCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_bpf_fastcall);

  // Constructors
  BPFFastCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  BPFFastCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::BPFFastCall; }
};

class CLANG_ABI BPFPreserveAccessIndexAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_preserve_access_index = 0,
    CXX11_clang_preserve_access_index = 1,
    C23_clang_preserve_access_index = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BPFPreserveAccessIndexAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFPreserveAccessIndexAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFPreserveAccessIndexAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_access_index);
  static BPFPreserveAccessIndexAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_access_index);

  // Constructors
  BPFPreserveAccessIndexAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  BPFPreserveAccessIndexAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::BPFPreserveAccessIndex; }
};

class CLANG_ABI BPFPreserveStaticOffsetAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_preserve_static_offset = 0,
    CXX11_clang_preserve_static_offset = 1,
    C23_clang_preserve_static_offset = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BPFPreserveStaticOffsetAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFPreserveStaticOffsetAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BPFPreserveStaticOffsetAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_static_offset);
  static BPFPreserveStaticOffsetAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_static_offset);

  // Constructors
  BPFPreserveStaticOffsetAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  BPFPreserveStaticOffsetAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::BPFPreserveStaticOffset; }
};

class CLANG_ABI BTFDeclTagAttr : public InheritableAttr {
unsigned bTFDeclTagLength;
char *bTFDeclTag;

public:
  enum Spelling {
    GNU_btf_decl_tag = 0,
    CXX11_clang_btf_decl_tag = 1,
    C23_clang_btf_decl_tag = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BTFDeclTagAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef BTFDeclTag, const AttributeCommonInfo &CommonInfo);
  static BTFDeclTagAttr *Create(ASTContext &Ctx, llvm::StringRef BTFDeclTag, const AttributeCommonInfo &CommonInfo);
  static BTFDeclTagAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef BTFDeclTag, SourceRange Range = {}, Spelling S = GNU_btf_decl_tag);
  static BTFDeclTagAttr *Create(ASTContext &Ctx, llvm::StringRef BTFDeclTag, SourceRange Range = {}, Spelling S = GNU_btf_decl_tag);

  // Constructors
  BTFDeclTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef BTFDeclTag
             );

  BTFDeclTagAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getBTFDeclTag() const {
    return llvm::StringRef(bTFDeclTag, bTFDeclTagLength);
  }
  unsigned getBTFDeclTagLength() const {
    return bTFDeclTagLength;
  }
  void setBTFDeclTag(ASTContext &C, llvm::StringRef S) {
    bTFDeclTagLength = S.size();
    this->bTFDeclTag = new (C, 1) char [bTFDeclTagLength];
    if (!S.empty())
      std::memcpy(this->bTFDeclTag, S.data(), bTFDeclTagLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::BTFDeclTag; }
};

class CLANG_ABI BTFTypeTagAttr : public TypeAttr {
unsigned bTFTypeTagLength;
char *bTFTypeTag;

public:
  enum Spelling {
    GNU_btf_type_tag = 0,
    CXX11_clang_btf_type_tag = 1,
    C23_clang_btf_type_tag = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BTFTypeTagAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef BTFTypeTag, const AttributeCommonInfo &CommonInfo);
  static BTFTypeTagAttr *Create(ASTContext &Ctx, llvm::StringRef BTFTypeTag, const AttributeCommonInfo &CommonInfo);
  static BTFTypeTagAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef BTFTypeTag, SourceRange Range = {}, Spelling S = GNU_btf_type_tag);
  static BTFTypeTagAttr *Create(ASTContext &Ctx, llvm::StringRef BTFTypeTag, SourceRange Range = {}, Spelling S = GNU_btf_type_tag);

  // Constructors
  BTFTypeTagAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef BTFTypeTag
             );

  BTFTypeTagAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getBTFTypeTag() const {
    return llvm::StringRef(bTFTypeTag, bTFTypeTagLength);
  }
  unsigned getBTFTypeTagLength() const {
    return bTFTypeTagLength;
  }
  void setBTFTypeTag(ASTContext &C, llvm::StringRef S) {
    bTFTypeTagLength = S.size();
    this->bTFTypeTag = new (C, 1) char [bTFTypeTagLength];
    if (!S.empty())
      std::memcpy(this->bTFTypeTag, S.data(), bTFTypeTagLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::BTFTypeTag; }
};

class CLANG_ABI BlockingAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_blocking = 0,
    CXX11_clang_blocking = 1,
    C23_clang_blocking = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BlockingAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BlockingAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static BlockingAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_blocking);
  static BlockingAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_blocking);

  // Constructors
  BlockingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  BlockingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Blocking; }
};

class CLANG_ABI BlocksAttr : public InheritableAttr {
public:
  enum BlockType {
    ByRef
  };
private:
  BlocksAttr::BlockType type;

public:
  enum Spelling {
    GNU_blocks = 0,
    CXX11_clang_blocks = 1,
    C23_clang_blocks = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BlocksAttr *CreateImplicit(ASTContext &Ctx, BlocksAttr::BlockType Type, const AttributeCommonInfo &CommonInfo);
  static BlocksAttr *Create(ASTContext &Ctx, BlocksAttr::BlockType Type, const AttributeCommonInfo &CommonInfo);
  static BlocksAttr *CreateImplicit(ASTContext &Ctx, BlocksAttr::BlockType Type, SourceRange Range = {}, Spelling S = GNU_blocks);
  static BlocksAttr *Create(ASTContext &Ctx, BlocksAttr::BlockType Type, SourceRange Range = {}, Spelling S = GNU_blocks);

  // Constructors
  BlocksAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , BlocksAttr::BlockType Type
             );

  BlocksAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  BlocksAttr::BlockType getType() const {
    return type;
  }

  static bool ConvertStrToBlockType(StringRef Val, BlocksAttr::BlockType &Out);
  static const char *ConvertBlockTypeToStr(BlocksAttr::BlockType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::Blocks; }
};

class CLANG_ABI BuiltinAttr : public InheritableAttr {
unsigned iD;

public:
  // Factory methods
  static BuiltinAttr *CreateImplicit(ASTContext &Ctx, unsigned ID, const AttributeCommonInfo &CommonInfo);
  static BuiltinAttr *Create(ASTContext &Ctx, unsigned ID, const AttributeCommonInfo &CommonInfo);
  static BuiltinAttr *CreateImplicit(ASTContext &Ctx, unsigned ID, SourceRange Range = {});
  static BuiltinAttr *Create(ASTContext &Ctx, unsigned ID, SourceRange Range = {});

  // Constructors
  BuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned ID
             );

  BuiltinAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getID() const {
    return iD;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Builtin; }
};

class CLANG_ABI BuiltinAliasAttr : public Attr {
IdentifierInfo * builtinName;

public:
  enum Spelling {
    CXX11_clang_builtin_alias = 0,
    C23_clang_builtin_alias = 1,
    GNU_clang_builtin_alias = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static BuiltinAliasAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BuiltinName, const AttributeCommonInfo &CommonInfo);
  static BuiltinAliasAttr *Create(ASTContext &Ctx, IdentifierInfo * BuiltinName, const AttributeCommonInfo &CommonInfo);
  static BuiltinAliasAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BuiltinName, SourceRange Range = {}, Spelling S = CXX11_clang_builtin_alias);
  static BuiltinAliasAttr *Create(ASTContext &Ctx, IdentifierInfo * BuiltinName, SourceRange Range = {}, Spelling S = CXX11_clang_builtin_alias);

  // Constructors
  BuiltinAliasAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * BuiltinName
             );

  BuiltinAliasAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  IdentifierInfo * getBuiltinName() const {
    return builtinName;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::BuiltinAlias; }
};

class CLANG_ABI C11NoReturnAttr : public InheritableAttr {
public:
  // Factory methods
  static C11NoReturnAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static C11NoReturnAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static C11NoReturnAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static C11NoReturnAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  C11NoReturnAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  C11NoReturnAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::C11NoReturn; }
};

class CLANG_ABI CDeclAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cdecl = 0,
    CXX11_gnu_cdecl = 1,
    C23_gnu_cdecl = 2,
    Keyword_cdecl = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CDeclAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CDeclAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CDeclAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cdecl);
  static CDeclAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cdecl);

  // Constructors
  CDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CDecl; }
};

class CLANG_ABI CFAuditedTransferAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cf_audited_transfer = 0,
    CXX11_clang_cf_audited_transfer = 1,
    C23_clang_cf_audited_transfer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFAuditedTransferAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFAuditedTransferAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFAuditedTransferAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_audited_transfer);
  static CFAuditedTransferAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_audited_transfer);

  // Constructors
  CFAuditedTransferAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFAuditedTransferAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFAuditedTransfer; }
};

class CLANG_ABI CFConsumedAttr : public InheritableParamAttr {
public:
  enum Spelling {
    GNU_cf_consumed = 0,
    CXX11_clang_cf_consumed = 1,
    C23_clang_cf_consumed = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFConsumedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFConsumedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFConsumedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_consumed);
  static CFConsumedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_consumed);

  // Constructors
  CFConsumedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFConsumedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFConsumed; }
};

class CLANG_ABI CFGuardAttr : public InheritableAttr {
public:
  enum GuardArg {
    nocf
  };
private:
  CFGuardAttr::GuardArg guard;

public:
  enum Spelling {
    Declspec_guard = 0,
    GNU_guard = 1,
    CXX11_clang_guard = 2,
    C23_clang_guard = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFGuardAttr *CreateImplicit(ASTContext &Ctx, CFGuardAttr::GuardArg Guard, const AttributeCommonInfo &CommonInfo);
  static CFGuardAttr *Create(ASTContext &Ctx, CFGuardAttr::GuardArg Guard, const AttributeCommonInfo &CommonInfo);
  static CFGuardAttr *CreateImplicit(ASTContext &Ctx, CFGuardAttr::GuardArg Guard, SourceRange Range = {}, Spelling S = Declspec_guard);
  static CFGuardAttr *Create(ASTContext &Ctx, CFGuardAttr::GuardArg Guard, SourceRange Range = {}, Spelling S = Declspec_guard);

  // Constructors
  CFGuardAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , CFGuardAttr::GuardArg Guard
             );

  CFGuardAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  CFGuardAttr::GuardArg getGuard() const {
    return guard;
  }

  static bool ConvertStrToGuardArg(StringRef Val, CFGuardAttr::GuardArg &Out);
  static const char *ConvertGuardArgToStr(CFGuardAttr::GuardArg Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::CFGuard; }
};

class CLANG_ABI CFICanonicalJumpTableAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cfi_canonical_jump_table = 0,
    CXX11_clang_cfi_canonical_jump_table = 1,
    C23_clang_cfi_canonical_jump_table = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFICanonicalJumpTableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFICanonicalJumpTableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFICanonicalJumpTableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cfi_canonical_jump_table);
  static CFICanonicalJumpTableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cfi_canonical_jump_table);

  // Constructors
  CFICanonicalJumpTableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFICanonicalJumpTableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFICanonicalJumpTable; }
};

class CLANG_ABI CFReturnsNotRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cf_returns_not_retained = 0,
    CXX11_clang_cf_returns_not_retained = 1,
    C23_clang_cf_returns_not_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFReturnsNotRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_returns_not_retained);
  static CFReturnsNotRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_returns_not_retained);

  // Constructors
  CFReturnsNotRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFReturnsNotRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFReturnsNotRetained; }
};

class CLANG_ABI CFReturnsRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cf_returns_retained = 0,
    CXX11_clang_cf_returns_retained = 1,
    C23_clang_cf_returns_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFReturnsRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_returns_retained);
  static CFReturnsRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_returns_retained);

  // Constructors
  CFReturnsRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFReturnsRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFReturnsRetained; }
};

class CLANG_ABI CFUnknownTransferAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cf_unknown_transfer = 0,
    CXX11_clang_cf_unknown_transfer = 1,
    C23_clang_cf_unknown_transfer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CFUnknownTransferAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFUnknownTransferAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CFUnknownTransferAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_unknown_transfer);
  static CFUnknownTransferAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cf_unknown_transfer);

  // Constructors
  CFUnknownTransferAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CFUnknownTransferAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CFUnknownTransfer; }
};

class CLANG_ABI CPUDispatchAttr : public InheritableAttr {
  unsigned cpus_Size;
  IdentifierInfo * *cpus_;

public:
  enum Spelling {
    GNU_cpu_dispatch = 0,
    CXX11_clang_cpu_dispatch = 1,
    C23_clang_cpu_dispatch = 2,
    Declspec_cpu_dispatch = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CPUDispatchAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, const AttributeCommonInfo &CommonInfo);
  static CPUDispatchAttr *Create(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, const AttributeCommonInfo &CommonInfo);
  static CPUDispatchAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, SourceRange Range = {}, Spelling S = GNU_cpu_dispatch);
  static CPUDispatchAttr *Create(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, SourceRange Range = {}, Spelling S = GNU_cpu_dispatch);

  // Constructors
  CPUDispatchAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * *Cpus, unsigned CpusSize
             );
  CPUDispatchAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CPUDispatchAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef IdentifierInfo ** cpus_iterator;
  cpus_iterator cpus_begin() const { return cpus_; }
  cpus_iterator cpus_end() const { return cpus_ + cpus_Size; }
  unsigned cpus_size() const { return cpus_Size; }
  llvm::iterator_range<cpus_iterator> cpus() const { return llvm::make_range(cpus_begin(), cpus_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::CPUDispatch; }
};

class CLANG_ABI CPUSpecificAttr : public InheritableAttr {
  unsigned cpus_Size;
  IdentifierInfo * *cpus_;

public:
  enum Spelling {
    GNU_cpu_specific = 0,
    CXX11_clang_cpu_specific = 1,
    C23_clang_cpu_specific = 2,
    Declspec_cpu_specific = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CPUSpecificAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, const AttributeCommonInfo &CommonInfo);
  static CPUSpecificAttr *Create(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, const AttributeCommonInfo &CommonInfo);
  static CPUSpecificAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, SourceRange Range = {}, Spelling S = GNU_cpu_specific);
  static CPUSpecificAttr *Create(ASTContext &Ctx, IdentifierInfo * *Cpus, unsigned CpusSize, SourceRange Range = {}, Spelling S = GNU_cpu_specific);

  // Constructors
  CPUSpecificAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * *Cpus, unsigned CpusSize
             );
  CPUSpecificAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CPUSpecificAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef IdentifierInfo ** cpus_iterator;
  cpus_iterator cpus_begin() const { return cpus_; }
  cpus_iterator cpus_end() const { return cpus_ + cpus_Size; }
  unsigned cpus_size() const { return cpus_Size; }
  llvm::iterator_range<cpus_iterator> cpus() const { return llvm::make_range(cpus_begin(), cpus_end()); }



    IdentifierInfo *getCPUName(unsigned Index) const {
      return *(cpus_begin() + Index);
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::CPUSpecific; }
};

class CLANG_ABI CUDAConstantAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_constant = 0,
    Declspec_constant = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDAConstantAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAConstantAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAConstantAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_constant);
  static CUDAConstantAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_constant);

  // Constructors
  CUDAConstantAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDAConstantAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAConstant; }
};

class CLANG_ABI CUDADeviceAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_device = 0,
    Declspec_device = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDADeviceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device);
  static CUDADeviceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device);

  // Constructors
  CUDADeviceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDADeviceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDADevice; }
};

class CLANG_ABI CUDADeviceBuiltinSurfaceTypeAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_device_builtin_surface_type = 0,
    Declspec_device_builtin_surface_type = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDADeviceBuiltinSurfaceTypeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceBuiltinSurfaceTypeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceBuiltinSurfaceTypeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device_builtin_surface_type);
  static CUDADeviceBuiltinSurfaceTypeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device_builtin_surface_type);

  // Constructors
  CUDADeviceBuiltinSurfaceTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDADeviceBuiltinSurfaceTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDADeviceBuiltinSurfaceType; }
};

class CLANG_ABI CUDADeviceBuiltinTextureTypeAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_device_builtin_texture_type = 0,
    Declspec_device_builtin_texture_type = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDADeviceBuiltinTextureTypeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceBuiltinTextureTypeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDADeviceBuiltinTextureTypeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device_builtin_texture_type);
  static CUDADeviceBuiltinTextureTypeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_device_builtin_texture_type);

  // Constructors
  CUDADeviceBuiltinTextureTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDADeviceBuiltinTextureTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDADeviceBuiltinTextureType; }
};

class CLANG_ABI CUDAGlobalAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_global = 0,
    Declspec_global = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDAGlobalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAGlobalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAGlobalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_global);
  static CUDAGlobalAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_global);

  // Constructors
  CUDAGlobalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDAGlobalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAGlobal; }
};

class CLANG_ABI CUDAGridConstantAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_grid_constant = 0,
    Declspec_grid_constant = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDAGridConstantAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAGridConstantAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAGridConstantAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_grid_constant);
  static CUDAGridConstantAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_grid_constant);

  // Constructors
  CUDAGridConstantAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDAGridConstantAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAGridConstant; }
};

class CLANG_ABI CUDAHostAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_host = 0,
    Declspec_host = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDAHostAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAHostAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAHostAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_host);
  static CUDAHostAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_host);

  // Constructors
  CUDAHostAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDAHostAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAHost; }
};

class CLANG_ABI CUDAInvalidTargetAttr : public InheritableAttr {
public:
  // Factory methods
  static CUDAInvalidTargetAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAInvalidTargetAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDAInvalidTargetAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static CUDAInvalidTargetAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  CUDAInvalidTargetAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDAInvalidTargetAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAInvalidTarget; }
};

class CLANG_ABI CUDALaunchBoundsAttr : public InheritableAttr {
Expr * maxThreads;

Expr * minBlocks;

Expr * maxBlocks;

public:
  enum Spelling {
    GNU_launch_bounds = 0,
    Declspec_launch_bounds = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDALaunchBoundsAttr *CreateImplicit(ASTContext &Ctx, Expr * MaxThreads, Expr * MinBlocks, Expr * MaxBlocks, const AttributeCommonInfo &CommonInfo);
  static CUDALaunchBoundsAttr *Create(ASTContext &Ctx, Expr * MaxThreads, Expr * MinBlocks, Expr * MaxBlocks, const AttributeCommonInfo &CommonInfo);
  static CUDALaunchBoundsAttr *CreateImplicit(ASTContext &Ctx, Expr * MaxThreads, Expr * MinBlocks, Expr * MaxBlocks, SourceRange Range = {}, Spelling S = GNU_launch_bounds);
  static CUDALaunchBoundsAttr *Create(ASTContext &Ctx, Expr * MaxThreads, Expr * MinBlocks, Expr * MaxBlocks, SourceRange Range = {}, Spelling S = GNU_launch_bounds);

  // Constructors
  CUDALaunchBoundsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * MaxThreads
              , Expr * MinBlocks
              , Expr * MaxBlocks
             );
  CUDALaunchBoundsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * MaxThreads
             );

  CUDALaunchBoundsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getMaxThreads() const {
    return maxThreads;
  }

  Expr * getMinBlocks() const {
    return minBlocks;
  }

  Expr * getMaxBlocks() const {
    return maxBlocks;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::CUDALaunchBounds; }
};

class CLANG_ABI CUDASharedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_shared = 0,
    Declspec_shared = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CUDASharedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDASharedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CUDASharedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_shared);
  static CUDASharedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_shared);

  // Constructors
  CUDASharedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CUDASharedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CUDAShared; }
};

class CLANG_ABI CXX11NoReturnAttr : public InheritableAttr {
public:
  enum Spelling {
    CXX11_noreturn = 0,
    C23_noreturn = 1,
    C23_Noreturn = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CXX11NoReturnAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CXX11NoReturnAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CXX11NoReturnAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_noreturn);
  static CXX11NoReturnAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_noreturn);

  // Constructors
  CXX11NoReturnAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CXX11NoReturnAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CXX11NoReturn; }
};

class CLANG_ABI CXXAssumeAttr : public StmtAttr {
Expr * assumption;

public:
  enum Spelling {
    CXX11_assume = 0,
    GNU_assume = 1,
    CXX11_clang_assume = 2,
    C23_clang_assume = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CXXAssumeAttr *CreateImplicit(ASTContext &Ctx, Expr * Assumption, const AttributeCommonInfo &CommonInfo);
  static CXXAssumeAttr *Create(ASTContext &Ctx, Expr * Assumption, const AttributeCommonInfo &CommonInfo);
  static CXXAssumeAttr *CreateImplicit(ASTContext &Ctx, Expr * Assumption, SourceRange Range = {}, Spelling S = CXX11_assume);
  static CXXAssumeAttr *Create(ASTContext &Ctx, Expr * Assumption, SourceRange Range = {}, Spelling S = CXX11_assume);

  // Constructors
  CXXAssumeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Assumption
             );

  CXXAssumeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getAssumption() const {
    return assumption;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::CXXAssume; }
};

class CLANG_ABI CallableWhenAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Unknown,
    Consumed,
    Unconsumed
  };
private:
  unsigned callableStates_Size;
  ConsumedState *callableStates_;

public:
  enum Spelling {
    GNU_callable_when = 0,
    CXX11_clang_callable_when = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CallableWhenAttr *CreateImplicit(ASTContext &Ctx, ConsumedState *CallableStates, unsigned CallableStatesSize, const AttributeCommonInfo &CommonInfo);
  static CallableWhenAttr *Create(ASTContext &Ctx, ConsumedState *CallableStates, unsigned CallableStatesSize, const AttributeCommonInfo &CommonInfo);
  static CallableWhenAttr *CreateImplicit(ASTContext &Ctx, ConsumedState *CallableStates, unsigned CallableStatesSize, SourceRange Range = {}, Spelling S = GNU_callable_when);
  static CallableWhenAttr *Create(ASTContext &Ctx, ConsumedState *CallableStates, unsigned CallableStatesSize, SourceRange Range = {}, Spelling S = GNU_callable_when);

  // Constructors
  CallableWhenAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ConsumedState *CallableStates, unsigned CallableStatesSize
             );
  CallableWhenAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CallableWhenAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef ConsumedState* callableStates_iterator;
  callableStates_iterator callableStates_begin() const { return callableStates_; }
  callableStates_iterator callableStates_end() const { return callableStates_ + callableStates_Size; }
  unsigned callableStates_size() const { return callableStates_Size; }
  llvm::iterator_range<callableStates_iterator> callableStates() const { return llvm::make_range(callableStates_begin(), callableStates_end()); }


  static bool ConvertStrToConsumedState(StringRef Val, CallableWhenAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(CallableWhenAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::CallableWhen; }
};

class CLANG_ABI CallbackAttr : public InheritableAttr {
  unsigned encoding_Size;
  int *encoding_;

public:
  enum Spelling {
    GNU_callback = 0,
    CXX11_clang_callback = 1,
    C23_clang_callback = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CallbackAttr *CreateImplicit(ASTContext &Ctx, int *Encoding, unsigned EncodingSize, const AttributeCommonInfo &CommonInfo);
  static CallbackAttr *Create(ASTContext &Ctx, int *Encoding, unsigned EncodingSize, const AttributeCommonInfo &CommonInfo);
  static CallbackAttr *CreateImplicit(ASTContext &Ctx, int *Encoding, unsigned EncodingSize, SourceRange Range = {}, Spelling S = GNU_callback);
  static CallbackAttr *Create(ASTContext &Ctx, int *Encoding, unsigned EncodingSize, SourceRange Range = {}, Spelling S = GNU_callback);

  // Constructors
  CallbackAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int *Encoding, unsigned EncodingSize
             );
  CallbackAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CallbackAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef int* encoding_iterator;
  encoding_iterator encoding_begin() const { return encoding_; }
  encoding_iterator encoding_end() const { return encoding_ + encoding_Size; }
  unsigned encoding_size() const { return encoding_Size; }
  llvm::iterator_range<encoding_iterator> encoding() const { return llvm::make_range(encoding_begin(), encoding_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::Callback; }
};

class CLANG_ABI CalledOnceAttr : public Attr {
public:
  enum Spelling {
    GNU_called_once = 0,
    CXX11_clang_called_once = 1,
    C23_clang_called_once = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CalledOnceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CalledOnceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CalledOnceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_called_once);
  static CalledOnceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_called_once);

  // Constructors
  CalledOnceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CalledOnceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CalledOnce; }
};

class CLANG_ABI CapabilityAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  enum Spelling {
    GNU_capability = 0,
    CXX11_clang_capability = 1,
    GNU_shared_capability = 2,
    CXX11_clang_shared_capability = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CapabilityAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static CapabilityAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static CapabilityAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {}, Spelling S = GNU_capability);
  static CapabilityAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {}, Spelling S = GNU_capability);

  // Constructors
  CapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  CapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3; }
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Capability; }
};

class CLANG_ABI CapturedRecordAttr : public InheritableAttr {
public:
  // Factory methods
  static CapturedRecordAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CapturedRecordAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CapturedRecordAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static CapturedRecordAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  CapturedRecordAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CapturedRecordAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CapturedRecord; }
};

class CLANG_ABI CarriesDependencyAttr : public InheritableParamAttr {
public:
  enum Spelling {
    GNU_carries_dependency = 0,
    CXX11_carries_dependency = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CarriesDependencyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CarriesDependencyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CarriesDependencyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_carries_dependency);
  static CarriesDependencyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_carries_dependency);

  // Constructors
  CarriesDependencyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CarriesDependencyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CarriesDependency; }
};

class CLANG_ABI CleanupAttr : public InheritableAttr {
FunctionDecl * functionDecl;

public:
  enum Spelling {
    GNU_cleanup = 0,
    CXX11_gnu_cleanup = 1,
    C23_gnu_cleanup = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CleanupAttr *CreateImplicit(ASTContext &Ctx, FunctionDecl * FunctionDecl, const AttributeCommonInfo &CommonInfo);
  static CleanupAttr *Create(ASTContext &Ctx, FunctionDecl * FunctionDecl, const AttributeCommonInfo &CommonInfo);
  static CleanupAttr *CreateImplicit(ASTContext &Ctx, FunctionDecl * FunctionDecl, SourceRange Range = {}, Spelling S = GNU_cleanup);
  static CleanupAttr *Create(ASTContext &Ctx, FunctionDecl * FunctionDecl, SourceRange Range = {}, Spelling S = GNU_cleanup);

  // Constructors
  CleanupAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , FunctionDecl * FunctionDecl
             );

  CleanupAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  FunctionDecl * getFunctionDecl() const {
    return functionDecl;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Cleanup; }
};

class CLANG_ABI ClspvLibclcBuiltinAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_clspv_libclc_builtin = 0,
    CXX11_clang_clspv_libclc_builtin = 1,
    C23_clang_clspv_libclc_builtin = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ClspvLibclcBuiltinAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ClspvLibclcBuiltinAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ClspvLibclcBuiltinAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_clspv_libclc_builtin);
  static ClspvLibclcBuiltinAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_clspv_libclc_builtin);

  // Constructors
  ClspvLibclcBuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ClspvLibclcBuiltinAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ClspvLibclcBuiltin; }
};

class CLANG_ABI CmseNSCallAttr : public TypeAttr {
public:
  // Factory methods
  static CmseNSCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CmseNSCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CmseNSCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static CmseNSCallAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  CmseNSCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CmseNSCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CmseNSCall; }
};

class CLANG_ABI CmseNSEntryAttr : public InheritableAttr {
public:
  // Factory methods
  static CmseNSEntryAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CmseNSEntryAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CmseNSEntryAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static CmseNSEntryAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  CmseNSEntryAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CmseNSEntryAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CmseNSEntry; }
};

class CLANG_ABI CodeAlignAttr : public StmtAttr {
Expr * alignment;

public:
  enum Spelling {
    GNU_code_align = 0,
    CXX11_clang_code_align = 1,
    C23_clang_code_align = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CodeAlignAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static CodeAlignAttr *Create(ASTContext &Ctx, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static CodeAlignAttr *CreateImplicit(ASTContext &Ctx, Expr * Alignment, SourceRange Range = {}, Spelling S = GNU_code_align);
  static CodeAlignAttr *Create(ASTContext &Ctx, Expr * Alignment, SourceRange Range = {}, Spelling S = GNU_code_align);

  // Constructors
  CodeAlignAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Alignment
             );

  CodeAlignAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getAlignment() const {
    return alignment;
  }


    static constexpr int MinimumAlignment = 1;
    static constexpr int MaximumAlignment = 4096;
  

  static bool classof(const Attr *A) { return A->getKind() == attr::CodeAlign; }
};

class CLANG_ABI CodeModelAttr : public InheritableAttr {
private:
  llvm::CodeModel::Model model;

public:
  enum Spelling {
    GNU_model = 0,
    CXX11_gnu_model = 1,
    C23_gnu_model = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CodeModelAttr *CreateImplicit(ASTContext &Ctx, llvm::CodeModel::Model Model, const AttributeCommonInfo &CommonInfo);
  static CodeModelAttr *Create(ASTContext &Ctx, llvm::CodeModel::Model Model, const AttributeCommonInfo &CommonInfo);
  static CodeModelAttr *CreateImplicit(ASTContext &Ctx, llvm::CodeModel::Model Model, SourceRange Range = {}, Spelling S = GNU_model);
  static CodeModelAttr *Create(ASTContext &Ctx, llvm::CodeModel::Model Model, SourceRange Range = {}, Spelling S = GNU_model);

  // Constructors
  CodeModelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::CodeModel::Model Model
             );

  CodeModelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::CodeModel::Model getModel() const {
    return model;
  }

  static bool ConvertStrToModel(StringRef Val, llvm::CodeModel::Model &Out);
  static const char *ConvertModelToStr(llvm::CodeModel::Model Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::CodeModel; }
};

class CLANG_ABI CodeSegAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static CodeSegAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static CodeSegAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static CodeSegAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static CodeSegAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  CodeSegAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  CodeSegAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::CodeSeg; }
};

class CLANG_ABI ColdAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_cold = 0,
    CXX11_gnu_cold = 1,
    C23_gnu_cold = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ColdAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ColdAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ColdAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cold);
  static ColdAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_cold);

  // Constructors
  ColdAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ColdAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Cold; }
};

class CLANG_ABI CommonAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_common = 0,
    CXX11_gnu_common = 1,
    C23_gnu_common = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CommonAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CommonAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CommonAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_common);
  static CommonAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_common);

  // Constructors
  CommonAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CommonAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Common; }
};

class CLANG_ABI ConstAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_const = 0,
    CXX11_gnu_const = 1,
    C23_gnu_const = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConstAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConstAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConstAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_const);
  static ConstAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_const);

  // Constructors
  ConstAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConstAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Const; }
};

class CLANG_ABI ConstInitAttr : public InheritableAttr {
public:
  enum Spelling {
    Keyword_constinit = 0,
    GNU_require_constant_initialization = 1,
    CXX11_clang_require_constant_initialization = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConstInitAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConstInitAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConstInitAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_constinit);
  static ConstInitAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_constinit);

  // Constructors
  ConstInitAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConstInitAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isConstinit() const { return getAttributeSpellingListIndex() == 0; }


  static bool classof(const Attr *A) { return A->getKind() == attr::ConstInit; }
};

class CLANG_ABI ConstructorAttr : public InheritableAttr {
int priority;

public:
  enum Spelling {
    GNU_constructor = 0,
    CXX11_gnu_constructor = 1,
    C23_gnu_constructor = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConstructorAttr *CreateImplicit(ASTContext &Ctx, int Priority, const AttributeCommonInfo &CommonInfo);
  static ConstructorAttr *Create(ASTContext &Ctx, int Priority, const AttributeCommonInfo &CommonInfo);
  static ConstructorAttr *CreateImplicit(ASTContext &Ctx, int Priority, SourceRange Range = {}, Spelling S = GNU_constructor);
  static ConstructorAttr *Create(ASTContext &Ctx, int Priority, SourceRange Range = {}, Spelling S = GNU_constructor);

  // Constructors
  ConstructorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Priority
             );
  ConstructorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConstructorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getPriority() const {
    return priority;
  }

  static const int DefaultPriority = 65535;



  static bool classof(const Attr *A) { return A->getKind() == attr::Constructor; }
};

class CLANG_ABI ConsumableAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Unknown,
    Consumed,
    Unconsumed
  };
private:
  ConsumableAttr::ConsumedState defaultState;

public:
  enum Spelling {
    GNU_consumable = 0,
    CXX11_clang_consumable = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConsumableAttr *CreateImplicit(ASTContext &Ctx, ConsumableAttr::ConsumedState DefaultState, const AttributeCommonInfo &CommonInfo);
  static ConsumableAttr *Create(ASTContext &Ctx, ConsumableAttr::ConsumedState DefaultState, const AttributeCommonInfo &CommonInfo);
  static ConsumableAttr *CreateImplicit(ASTContext &Ctx, ConsumableAttr::ConsumedState DefaultState, SourceRange Range = {}, Spelling S = GNU_consumable);
  static ConsumableAttr *Create(ASTContext &Ctx, ConsumableAttr::ConsumedState DefaultState, SourceRange Range = {}, Spelling S = GNU_consumable);

  // Constructors
  ConsumableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ConsumableAttr::ConsumedState DefaultState
             );

  ConsumableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ConsumableAttr::ConsumedState getDefaultState() const {
    return defaultState;
  }

  static bool ConvertStrToConsumedState(StringRef Val, ConsumableAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(ConsumableAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::Consumable; }
};

class CLANG_ABI ConsumableAutoCastAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_consumable_auto_cast_state = 0,
    CXX11_clang_consumable_auto_cast_state = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConsumableAutoCastAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConsumableAutoCastAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConsumableAutoCastAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_consumable_auto_cast_state);
  static ConsumableAutoCastAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_consumable_auto_cast_state);

  // Constructors
  ConsumableAutoCastAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConsumableAutoCastAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ConsumableAutoCast; }
};

class CLANG_ABI ConsumableSetOnReadAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_consumable_set_state_on_read = 0,
    CXX11_clang_consumable_set_state_on_read = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConsumableSetOnReadAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConsumableSetOnReadAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConsumableSetOnReadAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_consumable_set_state_on_read);
  static ConsumableSetOnReadAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_consumable_set_state_on_read);

  // Constructors
  ConsumableSetOnReadAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConsumableSetOnReadAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ConsumableSetOnRead; }
};

class CLANG_ABI ConvergentAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_convergent = 0,
    CXX11_clang_convergent = 1,
    C23_clang_convergent = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ConvergentAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConvergentAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ConvergentAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_convergent);
  static ConvergentAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_convergent);

  // Constructors
  ConvergentAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ConvergentAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Convergent; }
};

class CLANG_ABI CoroAwaitElidableAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_await_elidable = 0,
    CXX11_clang_coro_await_elidable = 1,
    C23_clang_coro_await_elidable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroAwaitElidableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroAwaitElidableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroAwaitElidableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_await_elidable);
  static CoroAwaitElidableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_await_elidable);

  // Constructors
  CoroAwaitElidableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroAwaitElidableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroAwaitElidable; }
};

class CLANG_ABI CoroAwaitElidableArgumentAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_await_elidable_argument = 0,
    CXX11_clang_coro_await_elidable_argument = 1,
    C23_clang_coro_await_elidable_argument = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroAwaitElidableArgumentAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroAwaitElidableArgumentAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroAwaitElidableArgumentAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_await_elidable_argument);
  static CoroAwaitElidableArgumentAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_await_elidable_argument);

  // Constructors
  CoroAwaitElidableArgumentAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroAwaitElidableArgumentAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroAwaitElidableArgument; }
};

class CLANG_ABI CoroDisableLifetimeBoundAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_disable_lifetimebound = 0,
    CXX11_clang_coro_disable_lifetimebound = 1,
    C23_clang_coro_disable_lifetimebound = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroDisableLifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroDisableLifetimeBoundAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroDisableLifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_disable_lifetimebound);
  static CoroDisableLifetimeBoundAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_disable_lifetimebound);

  // Constructors
  CoroDisableLifetimeBoundAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroDisableLifetimeBoundAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroDisableLifetimeBound; }
};

class CLANG_ABI CoroLifetimeBoundAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_lifetimebound = 0,
    CXX11_clang_coro_lifetimebound = 1,
    C23_clang_coro_lifetimebound = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroLifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroLifetimeBoundAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroLifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_lifetimebound);
  static CoroLifetimeBoundAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_lifetimebound);

  // Constructors
  CoroLifetimeBoundAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroLifetimeBoundAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroLifetimeBound; }
};

class CLANG_ABI CoroOnlyDestroyWhenCompleteAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_only_destroy_when_complete = 0,
    CXX11_clang_coro_only_destroy_when_complete = 1,
    C23_clang_coro_only_destroy_when_complete = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroOnlyDestroyWhenCompleteAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroOnlyDestroyWhenCompleteAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroOnlyDestroyWhenCompleteAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_only_destroy_when_complete);
  static CoroOnlyDestroyWhenCompleteAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_only_destroy_when_complete);

  // Constructors
  CoroOnlyDestroyWhenCompleteAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroOnlyDestroyWhenCompleteAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroOnlyDestroyWhenComplete; }
};

class CLANG_ABI CoroReturnTypeAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_return_type = 0,
    CXX11_clang_coro_return_type = 1,
    C23_clang_coro_return_type = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroReturnTypeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroReturnTypeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroReturnTypeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_return_type);
  static CoroReturnTypeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_return_type);

  // Constructors
  CoroReturnTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroReturnTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroReturnType; }
};

class CLANG_ABI CoroWrapperAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_coro_wrapper = 0,
    CXX11_clang_coro_wrapper = 1,
    C23_clang_coro_wrapper = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CoroWrapperAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroWrapperAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static CoroWrapperAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_wrapper);
  static CoroWrapperAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_coro_wrapper);

  // Constructors
  CoroWrapperAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  CoroWrapperAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::CoroWrapper; }
};

class CLANG_ABI CountedByAttr : public InheritableAttr {
Expr * count;

int nestedLevel;

public:
  enum Spelling {
    GNU_counted_by = 0,
    CXX11_clang_counted_by = 1,
    C23_clang_counted_by = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CountedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Count, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static CountedByAttr *Create(ASTContext &Ctx, Expr * Count, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static CountedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Count, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_counted_by);
  static CountedByAttr *Create(ASTContext &Ctx, Expr * Count, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_counted_by);

  // Constructors
  CountedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Count
              , int NestedLevel
             );
  CountedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Count
             );

  CountedByAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCount() const {
    return count;
  }

  int getNestedLevel() const {
    return nestedLevel;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::CountedBy; }
};

class CLANG_ABI CountedByOrNullAttr : public InheritableAttr {
Expr * count;

int nestedLevel;

public:
  enum Spelling {
    GNU_counted_by_or_null = 0,
    CXX11_clang_counted_by_or_null = 1,
    C23_clang_counted_by_or_null = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static CountedByOrNullAttr *CreateImplicit(ASTContext &Ctx, Expr * Count, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static CountedByOrNullAttr *Create(ASTContext &Ctx, Expr * Count, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static CountedByOrNullAttr *CreateImplicit(ASTContext &Ctx, Expr * Count, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_counted_by_or_null);
  static CountedByOrNullAttr *Create(ASTContext &Ctx, Expr * Count, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_counted_by_or_null);

  // Constructors
  CountedByOrNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Count
              , int NestedLevel
             );
  CountedByOrNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Count
             );

  CountedByOrNullAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCount() const {
    return count;
  }

  int getNestedLevel() const {
    return nestedLevel;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::CountedByOrNull; }
};

class CLANG_ABI DLLExportAttr : public InheritableAttr {
public:
  enum Spelling {
    Declspec_dllexport = 0,
    GNU_dllexport = 1,
    CXX11_gnu_dllexport = 2,
    C23_gnu_dllexport = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DLLExportAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLExportAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLExportAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_dllexport);
  static DLLExportAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_dllexport);

  // Constructors
  DLLExportAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DLLExportAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::DLLExport; }
};

class CLANG_ABI DLLExportStaticLocalAttr : public InheritableAttr {
public:
  // Factory methods
  static DLLExportStaticLocalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLExportStaticLocalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLExportStaticLocalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static DLLExportStaticLocalAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  DLLExportStaticLocalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DLLExportStaticLocalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::DLLExportStaticLocal; }
};

class CLANG_ABI DLLImportAttr : public InheritableAttr {
public:
  enum Spelling {
    Declspec_dllimport = 0,
    GNU_dllimport = 1,
    CXX11_gnu_dllimport = 2,
    C23_gnu_dllimport = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DLLImportAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLImportAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLImportAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_dllimport);
  static DLLImportAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_dllimport);

  // Constructors
  DLLImportAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DLLImportAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;

private:
  bool PropagatedToBaseTemplate = false;

public:
  void setPropagatedToBaseTemplate() { PropagatedToBaseTemplate = true; }
  bool wasPropagatedToBaseTemplate() { return PropagatedToBaseTemplate; }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::DLLImport; }
};

class CLANG_ABI DLLImportStaticLocalAttr : public InheritableAttr {
public:
  // Factory methods
  static DLLImportStaticLocalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLImportStaticLocalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DLLImportStaticLocalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static DLLImportStaticLocalAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  DLLImportStaticLocalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DLLImportStaticLocalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::DLLImportStaticLocal; }
};

class CLANG_ABI DeprecatedAttr : public InheritableAttr {
unsigned messageLength;
char *message;

unsigned replacementLength;
char *replacement;

public:
  enum Spelling {
    GNU_deprecated = 0,
    CXX11_gnu_deprecated = 1,
    C23_gnu_deprecated = 2,
    Declspec_deprecated = 3,
    CXX11_deprecated = 4,
    C23_deprecated = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DeprecatedAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, llvm::StringRef Replacement, const AttributeCommonInfo &CommonInfo);
  static DeprecatedAttr *Create(ASTContext &Ctx, llvm::StringRef Message, llvm::StringRef Replacement, const AttributeCommonInfo &CommonInfo);
  static DeprecatedAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, llvm::StringRef Replacement, SourceRange Range = {}, Spelling S = GNU_deprecated);
  static DeprecatedAttr *Create(ASTContext &Ctx, llvm::StringRef Message, llvm::StringRef Replacement, SourceRange Range = {}, Spelling S = GNU_deprecated);

  // Constructors
  DeprecatedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Message
              , llvm::StringRef Replacement
             );
  DeprecatedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DeprecatedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }

  llvm::StringRef getReplacement() const {
    return llvm::StringRef(replacement, replacementLength);
  }
  unsigned getReplacementLength() const {
    return replacementLength;
  }
  void setReplacement(ASTContext &C, llvm::StringRef S) {
    replacementLength = S.size();
    this->replacement = new (C, 1) char [replacementLength];
    if (!S.empty())
      std::memcpy(this->replacement, S.data(), replacementLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Deprecated; }
};

class CLANG_ABI DestructorAttr : public InheritableAttr {
int priority;

public:
  enum Spelling {
    GNU_destructor = 0,
    CXX11_gnu_destructor = 1,
    C23_gnu_destructor = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DestructorAttr *CreateImplicit(ASTContext &Ctx, int Priority, const AttributeCommonInfo &CommonInfo);
  static DestructorAttr *Create(ASTContext &Ctx, int Priority, const AttributeCommonInfo &CommonInfo);
  static DestructorAttr *CreateImplicit(ASTContext &Ctx, int Priority, SourceRange Range = {}, Spelling S = GNU_destructor);
  static DestructorAttr *Create(ASTContext &Ctx, int Priority, SourceRange Range = {}, Spelling S = GNU_destructor);

  // Constructors
  DestructorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Priority
             );
  DestructorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DestructorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getPriority() const {
    return priority;
  }

  static const int DefaultPriority = 65535;



  static bool classof(const Attr *A) { return A->getKind() == attr::Destructor; }
};

class CLANG_ABI DiagnoseAsBuiltinAttr : public InheritableAttr {
FunctionDecl * function;

  unsigned argIndices_Size;
  unsigned *argIndices_;

public:
  enum Spelling {
    GNU_diagnose_as_builtin = 0,
    CXX11_clang_diagnose_as_builtin = 1,
    C23_clang_diagnose_as_builtin = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DiagnoseAsBuiltinAttr *CreateImplicit(ASTContext &Ctx, FunctionDecl * Function, unsigned *ArgIndices, unsigned ArgIndicesSize, const AttributeCommonInfo &CommonInfo);
  static DiagnoseAsBuiltinAttr *Create(ASTContext &Ctx, FunctionDecl * Function, unsigned *ArgIndices, unsigned ArgIndicesSize, const AttributeCommonInfo &CommonInfo);
  static DiagnoseAsBuiltinAttr *CreateImplicit(ASTContext &Ctx, FunctionDecl * Function, unsigned *ArgIndices, unsigned ArgIndicesSize, SourceRange Range = {}, Spelling S = GNU_diagnose_as_builtin);
  static DiagnoseAsBuiltinAttr *Create(ASTContext &Ctx, FunctionDecl * Function, unsigned *ArgIndices, unsigned ArgIndicesSize, SourceRange Range = {}, Spelling S = GNU_diagnose_as_builtin);

  // Constructors
  DiagnoseAsBuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , FunctionDecl * Function
              , unsigned *ArgIndices, unsigned ArgIndicesSize
             );
  DiagnoseAsBuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , FunctionDecl * Function
             );

  DiagnoseAsBuiltinAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  FunctionDecl * getFunction() const {
    return function;
  }

  typedef unsigned* argIndices_iterator;
  argIndices_iterator argIndices_begin() const { return argIndices_; }
  argIndices_iterator argIndices_end() const { return argIndices_ + argIndices_Size; }
  unsigned argIndices_size() const { return argIndices_Size; }
  llvm::iterator_range<argIndices_iterator> argIndices() const { return llvm::make_range(argIndices_begin(), argIndices_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::DiagnoseAsBuiltin; }
};

class CLANG_ABI DiagnoseIfAttr : public InheritableAttr {
Expr * cond;

unsigned messageLength;
char *message;

public:
  enum DefaultSeverity {
    DS_error,
    DS_warning
  };
private:
  DiagnoseIfAttr::DefaultSeverity defaultSeverity;

unsigned warningGroupLength;
char *warningGroup;

bool argDependent;

NamedDecl * parent;

public:
  // Factory methods
  static DiagnoseIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, bool ArgDependent, NamedDecl * Parent, const AttributeCommonInfo &CommonInfo);
  static DiagnoseIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, bool ArgDependent, NamedDecl * Parent, const AttributeCommonInfo &CommonInfo);
  static DiagnoseIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, bool ArgDependent, NamedDecl * Parent, SourceRange Range = {});
  static DiagnoseIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, bool ArgDependent, NamedDecl * Parent, SourceRange Range = {});
  static DiagnoseIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, const AttributeCommonInfo &CommonInfo);
  static DiagnoseIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, const AttributeCommonInfo &CommonInfo);
  static DiagnoseIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, SourceRange Range = {});
  static DiagnoseIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, DiagnoseIfAttr::DefaultSeverity DefaultSeverity, llvm::StringRef WarningGroup, SourceRange Range = {});

  // Constructors
  DiagnoseIfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
              , llvm::StringRef Message
              , DiagnoseIfAttr::DefaultSeverity DefaultSeverity
              , llvm::StringRef WarningGroup
              , bool ArgDependent
              , NamedDecl * Parent
             );
  DiagnoseIfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
              , llvm::StringRef Message
              , DiagnoseIfAttr::DefaultSeverity DefaultSeverity
              , llvm::StringRef WarningGroup
             );
  DiagnoseIfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
              , llvm::StringRef Message
              , DiagnoseIfAttr::DefaultSeverity DefaultSeverity
             );

  DiagnoseIfAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCond() const {
    return cond;
  }

  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }

  DiagnoseIfAttr::DefaultSeverity getDefaultSeverity() const {
    return defaultSeverity;
  }

  static bool ConvertStrToDefaultSeverity(StringRef Val, DiagnoseIfAttr::DefaultSeverity &Out);
  static const char *ConvertDefaultSeverityToStr(DiagnoseIfAttr::DefaultSeverity Val);
  llvm::StringRef getWarningGroup() const {
    return llvm::StringRef(warningGroup, warningGroupLength);
  }
  unsigned getWarningGroupLength() const {
    return warningGroupLength;
  }
  void setWarningGroup(ASTContext &C, llvm::StringRef S) {
    warningGroupLength = S.size();
    this->warningGroup = new (C, 1) char [warningGroupLength];
    if (!S.empty())
      std::memcpy(this->warningGroup, S.data(), warningGroupLength);
  }

  bool getArgDependent() const {
    return argDependent;
  }

  NamedDecl * getParent() const {
    return parent;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::DiagnoseIf; }
};

class CLANG_ABI DisableSanitizerInstrumentationAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_disable_sanitizer_instrumentation = 0,
    CXX11_clang_disable_sanitizer_instrumentation = 1,
    C23_clang_disable_sanitizer_instrumentation = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DisableSanitizerInstrumentationAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DisableSanitizerInstrumentationAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DisableSanitizerInstrumentationAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_disable_sanitizer_instrumentation);
  static DisableSanitizerInstrumentationAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_disable_sanitizer_instrumentation);

  // Constructors
  DisableSanitizerInstrumentationAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DisableSanitizerInstrumentationAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::DisableSanitizerInstrumentation; }
};

class CLANG_ABI DisableTailCallsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_disable_tail_calls = 0,
    CXX11_clang_disable_tail_calls = 1,
    C23_clang_disable_tail_calls = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static DisableTailCallsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DisableTailCallsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static DisableTailCallsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_disable_tail_calls);
  static DisableTailCallsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_disable_tail_calls);

  // Constructors
  DisableTailCallsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  DisableTailCallsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::DisableTailCalls; }
};

class CLANG_ABI EmptyBasesAttr : public InheritableAttr {
public:
  // Factory methods
  static EmptyBasesAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static EmptyBasesAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static EmptyBasesAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static EmptyBasesAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  EmptyBasesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  EmptyBasesAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::EmptyBases; }
};

class CLANG_ABI EnableIfAttr : public InheritableAttr {
Expr * cond;

unsigned messageLength;
char *message;

public:
  // Factory methods
  static EnableIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static EnableIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static EnableIfAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, SourceRange Range = {});
  static EnableIfAttr *Create(ASTContext &Ctx, Expr * Cond, llvm::StringRef Message, SourceRange Range = {});

  // Constructors
  EnableIfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
              , llvm::StringRef Message
             );

  EnableIfAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCond() const {
    return cond;
  }

  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::EnableIf; }
};

class CLANG_ABI EnforceTCBAttr : public InheritableAttr {
unsigned tCBNameLength;
char *tCBName;

public:
  enum Spelling {
    GNU_enforce_tcb = 0,
    CXX11_clang_enforce_tcb = 1,
    C23_clang_enforce_tcb = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static EnforceTCBAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef TCBName, const AttributeCommonInfo &CommonInfo);
  static EnforceTCBAttr *Create(ASTContext &Ctx, llvm::StringRef TCBName, const AttributeCommonInfo &CommonInfo);
  static EnforceTCBAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef TCBName, SourceRange Range = {}, Spelling S = GNU_enforce_tcb);
  static EnforceTCBAttr *Create(ASTContext &Ctx, llvm::StringRef TCBName, SourceRange Range = {}, Spelling S = GNU_enforce_tcb);

  // Constructors
  EnforceTCBAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef TCBName
             );

  EnforceTCBAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getTCBName() const {
    return llvm::StringRef(tCBName, tCBNameLength);
  }
  unsigned getTCBNameLength() const {
    return tCBNameLength;
  }
  void setTCBName(ASTContext &C, llvm::StringRef S) {
    tCBNameLength = S.size();
    this->tCBName = new (C, 1) char [tCBNameLength];
    if (!S.empty())
      std::memcpy(this->tCBName, S.data(), tCBNameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::EnforceTCB; }
};

class CLANG_ABI EnforceTCBLeafAttr : public InheritableAttr {
unsigned tCBNameLength;
char *tCBName;

public:
  enum Spelling {
    GNU_enforce_tcb_leaf = 0,
    CXX11_clang_enforce_tcb_leaf = 1,
    C23_clang_enforce_tcb_leaf = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static EnforceTCBLeafAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef TCBName, const AttributeCommonInfo &CommonInfo);
  static EnforceTCBLeafAttr *Create(ASTContext &Ctx, llvm::StringRef TCBName, const AttributeCommonInfo &CommonInfo);
  static EnforceTCBLeafAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef TCBName, SourceRange Range = {}, Spelling S = GNU_enforce_tcb_leaf);
  static EnforceTCBLeafAttr *Create(ASTContext &Ctx, llvm::StringRef TCBName, SourceRange Range = {}, Spelling S = GNU_enforce_tcb_leaf);

  // Constructors
  EnforceTCBLeafAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef TCBName
             );

  EnforceTCBLeafAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getTCBName() const {
    return llvm::StringRef(tCBName, tCBNameLength);
  }
  unsigned getTCBNameLength() const {
    return tCBNameLength;
  }
  void setTCBName(ASTContext &C, llvm::StringRef S) {
    tCBNameLength = S.size();
    this->tCBName = new (C, 1) char [tCBNameLength];
    if (!S.empty())
      std::memcpy(this->tCBName, S.data(), tCBNameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::EnforceTCBLeaf; }
};

class CLANG_ABI EnumExtensibilityAttr : public InheritableAttr {
public:
  enum Kind {
    Closed,
    Open
  };
private:
  EnumExtensibilityAttr::Kind extensibility;

public:
  enum Spelling {
    GNU_enum_extensibility = 0,
    CXX11_clang_enum_extensibility = 1,
    C23_clang_enum_extensibility = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static EnumExtensibilityAttr *CreateImplicit(ASTContext &Ctx, EnumExtensibilityAttr::Kind Extensibility, const AttributeCommonInfo &CommonInfo);
  static EnumExtensibilityAttr *Create(ASTContext &Ctx, EnumExtensibilityAttr::Kind Extensibility, const AttributeCommonInfo &CommonInfo);
  static EnumExtensibilityAttr *CreateImplicit(ASTContext &Ctx, EnumExtensibilityAttr::Kind Extensibility, SourceRange Range = {}, Spelling S = GNU_enum_extensibility);
  static EnumExtensibilityAttr *Create(ASTContext &Ctx, EnumExtensibilityAttr::Kind Extensibility, SourceRange Range = {}, Spelling S = GNU_enum_extensibility);

  // Constructors
  EnumExtensibilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , EnumExtensibilityAttr::Kind Extensibility
             );

  EnumExtensibilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  EnumExtensibilityAttr::Kind getExtensibility() const {
    return extensibility;
  }

  static bool ConvertStrToKind(StringRef Val, EnumExtensibilityAttr::Kind &Out);
  static const char *ConvertKindToStr(EnumExtensibilityAttr::Kind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::EnumExtensibility; }
};

class CLANG_ABI ErrorAttr : public InheritableAttr {
unsigned userDiagnosticLength;
char *userDiagnostic;

public:
  enum Spelling {
    GNU_error = 0,
    CXX11_gnu_error = 1,
    C23_gnu_error = 2,
    GNU_warning = 3,
    CXX11_gnu_warning = 4,
    C23_gnu_warning = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ErrorAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef UserDiagnostic, const AttributeCommonInfo &CommonInfo);
  static ErrorAttr *Create(ASTContext &Ctx, llvm::StringRef UserDiagnostic, const AttributeCommonInfo &CommonInfo);
  static ErrorAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef UserDiagnostic, SourceRange Range = {}, Spelling S = GNU_error);
  static ErrorAttr *Create(ASTContext &Ctx, llvm::StringRef UserDiagnostic, SourceRange Range = {}, Spelling S = GNU_error);

  // Constructors
  ErrorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef UserDiagnostic
             );

  ErrorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isError() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1 ||
    getAttributeSpellingListIndex() == 2; }
  bool isWarning() const { return getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }
  llvm::StringRef getUserDiagnostic() const {
    return llvm::StringRef(userDiagnostic, userDiagnosticLength);
  }
  unsigned getUserDiagnosticLength() const {
    return userDiagnosticLength;
  }
  void setUserDiagnostic(ASTContext &C, llvm::StringRef S) {
    userDiagnosticLength = S.size();
    this->userDiagnostic = new (C, 1) char [userDiagnosticLength];
    if (!S.empty())
      std::memcpy(this->userDiagnostic, S.data(), userDiagnosticLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Error; }
};

class CLANG_ABI ExcludeFromExplicitInstantiationAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_exclude_from_explicit_instantiation = 0,
    CXX11_clang_exclude_from_explicit_instantiation = 1,
    C23_clang_exclude_from_explicit_instantiation = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ExcludeFromExplicitInstantiationAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ExcludeFromExplicitInstantiationAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ExcludeFromExplicitInstantiationAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_exclude_from_explicit_instantiation);
  static ExcludeFromExplicitInstantiationAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_exclude_from_explicit_instantiation);

  // Constructors
  ExcludeFromExplicitInstantiationAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ExcludeFromExplicitInstantiationAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ExcludeFromExplicitInstantiation; }
};

class CLANG_ABI ExplicitInitAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_require_explicit_initialization = 0,
    CXX11_clang_require_explicit_initialization = 1,
    C23_clang_require_explicit_initialization = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ExplicitInitAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ExplicitInitAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ExplicitInitAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_require_explicit_initialization);
  static ExplicitInitAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_require_explicit_initialization);

  // Constructors
  ExplicitInitAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ExplicitInitAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ExplicitInit; }
};

class CLANG_ABI ExtVectorTypeAttr : public TypeAttr {
Expr * numElements;

public:
  enum Spelling {
    GNU_ext_vector_type = 0,
    CXX11_clang_ext_vector_type = 1,
    C23_clang_ext_vector_type = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ExtVectorTypeAttr *CreateImplicit(ASTContext &Ctx, Expr * NumElements, const AttributeCommonInfo &CommonInfo);
  static ExtVectorTypeAttr *Create(ASTContext &Ctx, Expr * NumElements, const AttributeCommonInfo &CommonInfo);
  static ExtVectorTypeAttr *CreateImplicit(ASTContext &Ctx, Expr * NumElements, SourceRange Range = {}, Spelling S = GNU_ext_vector_type);
  static ExtVectorTypeAttr *Create(ASTContext &Ctx, Expr * NumElements, SourceRange Range = {}, Spelling S = GNU_ext_vector_type);

  // Constructors
  ExtVectorTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * NumElements
             );

  ExtVectorTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getNumElements() const {
    return numElements;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ExtVectorType; }
};

class CLANG_ABI ExternalSourceSymbolAttr : public InheritableAttr {
unsigned languageLength;
char *language;

unsigned definedInLength;
char *definedIn;

bool generatedDeclaration;

unsigned uSRLength;
char *uSR;

public:
  enum Spelling {
    GNU_external_source_symbol = 0,
    CXX11_clang_external_source_symbol = 1,
    C23_clang_external_source_symbol = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ExternalSourceSymbolAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Language, llvm::StringRef DefinedIn, bool GeneratedDeclaration, llvm::StringRef USR, const AttributeCommonInfo &CommonInfo);
  static ExternalSourceSymbolAttr *Create(ASTContext &Ctx, llvm::StringRef Language, llvm::StringRef DefinedIn, bool GeneratedDeclaration, llvm::StringRef USR, const AttributeCommonInfo &CommonInfo);
  static ExternalSourceSymbolAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Language, llvm::StringRef DefinedIn, bool GeneratedDeclaration, llvm::StringRef USR, SourceRange Range = {}, Spelling S = GNU_external_source_symbol);
  static ExternalSourceSymbolAttr *Create(ASTContext &Ctx, llvm::StringRef Language, llvm::StringRef DefinedIn, bool GeneratedDeclaration, llvm::StringRef USR, SourceRange Range = {}, Spelling S = GNU_external_source_symbol);

  // Constructors
  ExternalSourceSymbolAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Language
              , llvm::StringRef DefinedIn
              , bool GeneratedDeclaration
              , llvm::StringRef USR
             );
  ExternalSourceSymbolAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ExternalSourceSymbolAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getLanguage() const {
    return llvm::StringRef(language, languageLength);
  }
  unsigned getLanguageLength() const {
    return languageLength;
  }
  void setLanguage(ASTContext &C, llvm::StringRef S) {
    languageLength = S.size();
    this->language = new (C, 1) char [languageLength];
    if (!S.empty())
      std::memcpy(this->language, S.data(), languageLength);
  }

  llvm::StringRef getDefinedIn() const {
    return llvm::StringRef(definedIn, definedInLength);
  }
  unsigned getDefinedInLength() const {
    return definedInLength;
  }
  void setDefinedIn(ASTContext &C, llvm::StringRef S) {
    definedInLength = S.size();
    this->definedIn = new (C, 1) char [definedInLength];
    if (!S.empty())
      std::memcpy(this->definedIn, S.data(), definedInLength);
  }

  bool getGeneratedDeclaration() const {
    return generatedDeclaration;
  }

  llvm::StringRef getUSR() const {
    return llvm::StringRef(uSR, uSRLength);
  }
  unsigned getUSRLength() const {
    return uSRLength;
  }
  void setUSR(ASTContext &C, llvm::StringRef S) {
    uSRLength = S.size();
    this->uSR = new (C, 1) char [uSRLength];
    if (!S.empty())
      std::memcpy(this->uSR, S.data(), uSRLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ExternalSourceSymbol; }
};

class CLANG_ABI FallThroughAttr : public StmtAttr {
public:
  enum Spelling {
    CXX11_fallthrough = 0,
    C23_fallthrough = 1,
    CXX11_clang_fallthrough = 2,
    GNU_fallthrough = 3,
    CXX11_gnu_fallthrough = 4,
    C23_gnu_fallthrough = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FallThroughAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FallThroughAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FallThroughAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_fallthrough);
  static FallThroughAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_fallthrough);

  // Constructors
  FallThroughAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  FallThroughAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::FallThrough; }
};

class CLANG_ABI FastCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_fastcall = 0,
    CXX11_gnu_fastcall = 1,
    C23_gnu_fastcall = 2,
    Keyword_fastcall = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FastCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FastCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FastCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_fastcall);
  static FastCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_fastcall);

  // Constructors
  FastCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  FastCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::FastCall; }
};

class CLANG_ABI FinalAttr : public InheritableAttr {
public:
  enum Spelling {
    Keyword_final = 0,
    Keyword_sealed = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FinalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FinalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FinalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_final);
  static FinalAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_final);

  // Constructors
  FinalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  FinalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isSpelledAsSealed() const { return getAttributeSpellingListIndex() == 1; }


  static bool classof(const Attr *A) { return A->getKind() == attr::Final; }
};

class CLANG_ABI FlagEnumAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_flag_enum = 0,
    CXX11_clang_flag_enum = 1,
    C23_clang_flag_enum = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FlagEnumAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FlagEnumAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FlagEnumAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_flag_enum);
  static FlagEnumAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_flag_enum);

  // Constructors
  FlagEnumAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  FlagEnumAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::FlagEnum; }
};

class CLANG_ABI FlattenAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_flatten = 0,
    CXX11_gnu_flatten = 1,
    C23_gnu_flatten = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FlattenAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FlattenAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static FlattenAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_flatten);
  static FlattenAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_flatten);

  // Constructors
  FlattenAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  FlattenAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Flatten; }
};

class CLANG_ABI FormatAttr : public InheritableAttr {
IdentifierInfo * type;

int formatIdx;

int firstArg;

public:
  enum Spelling {
    GNU_format = 0,
    CXX11_gnu_format = 1,
    C23_gnu_format = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FormatAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, int FirstArg, const AttributeCommonInfo &CommonInfo);
  static FormatAttr *Create(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, int FirstArg, const AttributeCommonInfo &CommonInfo);
  static FormatAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, int FirstArg, SourceRange Range = {}, Spelling S = GNU_format);
  static FormatAttr *Create(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, int FirstArg, SourceRange Range = {}, Spelling S = GNU_format);

  // Constructors
  FormatAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Type
              , int FormatIdx
              , int FirstArg
             );

  FormatAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getType() const {
    return type;
  }

  int getFormatIdx() const {
    return formatIdx;
  }

  int getFirstArg() const {
    return firstArg;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Format; }
};

class CLANG_ABI FormatArgAttr : public InheritableAttr {
ParamIdx formatIdx;

public:
  enum Spelling {
    GNU_format_arg = 0,
    CXX11_gnu_format_arg = 1,
    C23_gnu_format_arg = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FormatArgAttr *CreateImplicit(ASTContext &Ctx, ParamIdx FormatIdx, const AttributeCommonInfo &CommonInfo);
  static FormatArgAttr *Create(ASTContext &Ctx, ParamIdx FormatIdx, const AttributeCommonInfo &CommonInfo);
  static FormatArgAttr *CreateImplicit(ASTContext &Ctx, ParamIdx FormatIdx, SourceRange Range = {}, Spelling S = GNU_format_arg);
  static FormatArgAttr *Create(ASTContext &Ctx, ParamIdx FormatIdx, SourceRange Range = {}, Spelling S = GNU_format_arg);

  // Constructors
  FormatArgAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamIdx FormatIdx
             );

  FormatArgAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ParamIdx getFormatIdx() const {
    return formatIdx;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::FormatArg; }
};

class CLANG_ABI FormatMatchesAttr : public InheritableAttr {
IdentifierInfo * type;

int formatIdx;

Expr * expectedFormat;

public:
  enum Spelling {
    GNU_format_matches = 0,
    CXX11_gnu_format_matches = 1,
    C23_gnu_format_matches = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FormatMatchesAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, Expr * ExpectedFormat, const AttributeCommonInfo &CommonInfo);
  static FormatMatchesAttr *Create(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, Expr * ExpectedFormat, const AttributeCommonInfo &CommonInfo);
  static FormatMatchesAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, Expr * ExpectedFormat, SourceRange Range = {}, Spelling S = GNU_format_matches);
  static FormatMatchesAttr *Create(ASTContext &Ctx, IdentifierInfo * Type, int FormatIdx, Expr * ExpectedFormat, SourceRange Range = {}, Spelling S = GNU_format_matches);

  // Constructors
  FormatMatchesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Type
              , int FormatIdx
              , Expr * ExpectedFormat
             );

  FormatMatchesAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getType() const {
    return type;
  }

  int getFormatIdx() const {
    return formatIdx;
  }

  Expr * getExpectedFormat() const {
    return expectedFormat;
  }


    StringLiteral *getFormatString() const;
  

  static bool classof(const Attr *A) { return A->getKind() == attr::FormatMatches; }
};

class CLANG_ABI FunctionReturnThunksAttr : public InheritableAttr {
public:
  enum Kind {
    Keep,
    Extern
  };
private:
  FunctionReturnThunksAttr::Kind thunkType;

public:
  enum Spelling {
    GNU_function_return = 0,
    CXX11_gnu_function_return = 1,
    C23_gnu_function_return = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static FunctionReturnThunksAttr *CreateImplicit(ASTContext &Ctx, FunctionReturnThunksAttr::Kind ThunkType, const AttributeCommonInfo &CommonInfo);
  static FunctionReturnThunksAttr *Create(ASTContext &Ctx, FunctionReturnThunksAttr::Kind ThunkType, const AttributeCommonInfo &CommonInfo);
  static FunctionReturnThunksAttr *CreateImplicit(ASTContext &Ctx, FunctionReturnThunksAttr::Kind ThunkType, SourceRange Range = {}, Spelling S = GNU_function_return);
  static FunctionReturnThunksAttr *Create(ASTContext &Ctx, FunctionReturnThunksAttr::Kind ThunkType, SourceRange Range = {}, Spelling S = GNU_function_return);

  // Constructors
  FunctionReturnThunksAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , FunctionReturnThunksAttr::Kind ThunkType
             );

  FunctionReturnThunksAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  FunctionReturnThunksAttr::Kind getThunkType() const {
    return thunkType;
  }

  static bool ConvertStrToKind(StringRef Val, FunctionReturnThunksAttr::Kind &Out);
  static const char *ConvertKindToStr(FunctionReturnThunksAttr::Kind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::FunctionReturnThunks; }
};

class CLANG_ABI GNUInlineAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_gnu_inline = 0,
    CXX11_gnu_gnu_inline = 1,
    C23_gnu_gnu_inline = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static GNUInlineAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static GNUInlineAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static GNUInlineAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_gnu_inline);
  static GNUInlineAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_gnu_inline);

  // Constructors
  GNUInlineAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  GNUInlineAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::GNUInline; }
};

class CLANG_ABI GuardedByAttr : public InheritableAttr {
Expr * arg;

public:
  // Factory methods
  static GuardedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static GuardedByAttr *Create(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static GuardedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});
  static GuardedByAttr *Create(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});

  // Constructors
  GuardedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Arg
             );

  GuardedByAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getArg() const {
    return arg;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::GuardedBy; }
};

class CLANG_ABI GuardedVarAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_guarded_var = 0,
    CXX11_clang_guarded_var = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static GuardedVarAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static GuardedVarAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static GuardedVarAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_guarded_var);
  static GuardedVarAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_guarded_var);

  // Constructors
  GuardedVarAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  GuardedVarAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::GuardedVar; }
};

class CLANG_ABI HIPManagedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_managed = 0,
    Declspec_managed = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HIPManagedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HIPManagedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HIPManagedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_managed);
  static HIPManagedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_managed);

  // Constructors
  HIPManagedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HIPManagedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HIPManaged; }
};

class CLANG_ABI HLSLContainedTypeAttr : public TypeAttr {
TypeSourceInfo * type;

public:
  // Factory methods
  static HLSLContainedTypeAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Type, const AttributeCommonInfo &CommonInfo);
  static HLSLContainedTypeAttr *Create(ASTContext &Ctx, TypeSourceInfo * Type, const AttributeCommonInfo &CommonInfo);
  static HLSLContainedTypeAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Type, SourceRange Range = {});
  static HLSLContainedTypeAttr *Create(ASTContext &Ctx, TypeSourceInfo * Type, SourceRange Range = {});

  // Constructors
  HLSLContainedTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * Type
             );

  HLSLContainedTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getType() const {
    return type->getType();
  }  TypeSourceInfo * getTypeLoc() const {
    return type;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLContainedType; }
};

class CLANG_ABI HLSLControlFlowHintAttr : public StmtAttr {
public:
  enum Spelling {
    Microsoft_branch = 0,
    Microsoft_flatten = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HLSLControlFlowHintAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLControlFlowHintAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLControlFlowHintAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Microsoft_branch);
  static HLSLControlFlowHintAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Microsoft_branch);

  // Constructors
  HLSLControlFlowHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLControlFlowHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLControlFlowHint; }
};

class CLANG_ABI HLSLGroupSharedAddressSpaceAttr : public TypeAttr {
public:
  // Factory methods
  static HLSLGroupSharedAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLGroupSharedAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLGroupSharedAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLGroupSharedAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLGroupSharedAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLGroupSharedAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLGroupSharedAddressSpace; }
};

class CLANG_ABI HLSLLoopHintAttr : public StmtAttr {
unsigned directive;

public:
  enum Spelling {
    Microsoft_unroll = 0,
    Microsoft_loop = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HLSLLoopHintAttr *CreateImplicit(ASTContext &Ctx, unsigned Directive, const AttributeCommonInfo &CommonInfo);
  static HLSLLoopHintAttr *Create(ASTContext &Ctx, unsigned Directive, const AttributeCommonInfo &CommonInfo);
  static HLSLLoopHintAttr *CreateImplicit(ASTContext &Ctx, unsigned Directive, SourceRange Range = {}, Spelling S = Microsoft_unroll);
  static HLSLLoopHintAttr *Create(ASTContext &Ctx, unsigned Directive, SourceRange Range = {}, Spelling S = Microsoft_unroll);

  // Constructors
  HLSLLoopHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Directive
             );
  HLSLLoopHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLLoopHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  unsigned getDirective() const {
    return directive;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLLoopHint; }
};

class CLANG_ABI HLSLNumThreadsAttr : public InheritableAttr {
int x;

int y;

int z;

public:
  // Factory methods
  static HLSLNumThreadsAttr *CreateImplicit(ASTContext &Ctx, int X, int Y, int Z, const AttributeCommonInfo &CommonInfo);
  static HLSLNumThreadsAttr *Create(ASTContext &Ctx, int X, int Y, int Z, const AttributeCommonInfo &CommonInfo);
  static HLSLNumThreadsAttr *CreateImplicit(ASTContext &Ctx, int X, int Y, int Z, SourceRange Range = {});
  static HLSLNumThreadsAttr *Create(ASTContext &Ctx, int X, int Y, int Z, SourceRange Range = {});

  // Constructors
  HLSLNumThreadsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int X
              , int Y
              , int Z
             );

  HLSLNumThreadsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getX() const {
    return x;
  }

  int getY() const {
    return y;
  }

  int getZ() const {
    return z;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLNumThreads; }
};

class CLANG_ABI HLSLPackOffsetAttr : public HLSLAnnotationAttr {
int subcomponent;

int component;

public:
  // Factory methods
  static HLSLPackOffsetAttr *CreateImplicit(ASTContext &Ctx, int Subcomponent, int Component, const AttributeCommonInfo &CommonInfo);
  static HLSLPackOffsetAttr *Create(ASTContext &Ctx, int Subcomponent, int Component, const AttributeCommonInfo &CommonInfo);
  static HLSLPackOffsetAttr *CreateImplicit(ASTContext &Ctx, int Subcomponent, int Component, SourceRange Range = {});
  static HLSLPackOffsetAttr *Create(ASTContext &Ctx, int Subcomponent, int Component, SourceRange Range = {});

  // Constructors
  HLSLPackOffsetAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Subcomponent
              , int Component
             );

  HLSLPackOffsetAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getSubcomponent() const {
    return subcomponent;
  }

  int getComponent() const {
    return component;
  }


  unsigned getOffsetInBytes() {
    return subcomponent * 16 + component * 4;
  }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLPackOffset; }
};

class CLANG_ABI HLSLParamModifierAttr : public ParameterABIAttr {
bool mergedSpelling;

public:
  enum Spelling {
    Keyword_in = 0,
    Keyword_inout = 1,
    Keyword_out = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HLSLParamModifierAttr *CreateImplicit(ASTContext &Ctx, bool MergedSpelling, const AttributeCommonInfo &CommonInfo);
  static HLSLParamModifierAttr *Create(ASTContext &Ctx, bool MergedSpelling, const AttributeCommonInfo &CommonInfo);
  static HLSLParamModifierAttr *CreateImplicit(ASTContext &Ctx, bool MergedSpelling, SourceRange Range = {}, Spelling S = Keyword_in);
  static HLSLParamModifierAttr *Create(ASTContext &Ctx, bool MergedSpelling, SourceRange Range = {}, Spelling S = Keyword_in);
  static HLSLParamModifierAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLParamModifierAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLParamModifierAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_in);
  static HLSLParamModifierAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_in);

  // Constructors
  HLSLParamModifierAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , bool MergedSpelling
             );
  HLSLParamModifierAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLParamModifierAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isIn() const { return getAttributeSpellingListIndex() == 0; }
  bool isInOut() const { return getAttributeSpellingListIndex() == 1; }
  bool isOut() const { return getAttributeSpellingListIndex() == 2; }
  bool isAnyOut() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 1; }
  bool isAnyIn() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1; }
  bool getMergedSpelling() const {
    return mergedSpelling;
  }

  static const bool DefaultMergedSpelling = false;



  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLParamModifier; }
};

class CLANG_ABI HLSLROVAttr : public TypeAttr {
public:
  // Factory methods
  static HLSLROVAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLROVAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLROVAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLROVAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLROVAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLROVAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLROV; }
};

class CLANG_ABI HLSLRawBufferAttr : public TypeAttr {
public:
  // Factory methods
  static HLSLRawBufferAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLRawBufferAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLRawBufferAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLRawBufferAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLRawBufferAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLRawBufferAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLRawBuffer; }
};

class CLANG_ABI HLSLResourceBindingAttr : public InheritableAttr {
unsigned slotLength;
char *slot;

unsigned spaceLength;
char *space;

public:
  // Factory methods
  static HLSLResourceBindingAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Slot, llvm::StringRef Space, const AttributeCommonInfo &CommonInfo);
  static HLSLResourceBindingAttr *Create(ASTContext &Ctx, llvm::StringRef Slot, llvm::StringRef Space, const AttributeCommonInfo &CommonInfo);
  static HLSLResourceBindingAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Slot, llvm::StringRef Space, SourceRange Range = {});
  static HLSLResourceBindingAttr *Create(ASTContext &Ctx, llvm::StringRef Slot, llvm::StringRef Space, SourceRange Range = {});

  // Constructors
  HLSLResourceBindingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Slot
              , llvm::StringRef Space
             );
  HLSLResourceBindingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Slot
             );

  HLSLResourceBindingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getSlot() const {
    return llvm::StringRef(slot, slotLength);
  }
  unsigned getSlotLength() const {
    return slotLength;
  }
  void setSlot(ASTContext &C, llvm::StringRef S) {
    slotLength = S.size();
    this->slot = new (C, 1) char [slotLength];
    if (!S.empty())
      std::memcpy(this->slot, S.data(), slotLength);
  }

  llvm::StringRef getSpace() const {
    return llvm::StringRef(space, spaceLength);
  }
  unsigned getSpaceLength() const {
    return spaceLength;
  }
  void setSpace(ASTContext &C, llvm::StringRef S) {
    spaceLength = S.size();
    this->space = new (C, 1) char [spaceLength];
    if (!S.empty())
      std::memcpy(this->space, S.data(), spaceLength);
  }


  public:
      enum class RegisterType : unsigned { SRV, UAV, CBuffer, Sampler, C, I };

  private:
      RegisterType RegType;
      unsigned SlotNumber;
      unsigned SpaceNumber;

  public:
      void setBinding(RegisterType RT, unsigned SlotNum, unsigned SpaceNum) {
        RegType = RT;
        SlotNumber = SlotNum;
        SpaceNumber = SpaceNum;
      }
      RegisterType getRegisterType() const {
        return RegType;
      }
      unsigned getSlotNumber() const {
        return SlotNumber;
      }
      unsigned getSpaceNumber() const {
        return SpaceNumber;
      }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLResourceBinding; }
};

class CLANG_ABI HLSLResourceClassAttr : public TypeAttr {
private:
  llvm::hlsl::ResourceClass resourceClass;

public:
  // Factory methods
  static HLSLResourceClassAttr *CreateImplicit(ASTContext &Ctx, llvm::hlsl::ResourceClass ResourceClass, const AttributeCommonInfo &CommonInfo);
  static HLSLResourceClassAttr *Create(ASTContext &Ctx, llvm::hlsl::ResourceClass ResourceClass, const AttributeCommonInfo &CommonInfo);
  static HLSLResourceClassAttr *CreateImplicit(ASTContext &Ctx, llvm::hlsl::ResourceClass ResourceClass, SourceRange Range = {});
  static HLSLResourceClassAttr *Create(ASTContext &Ctx, llvm::hlsl::ResourceClass ResourceClass, SourceRange Range = {});

  // Constructors
  HLSLResourceClassAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::hlsl::ResourceClass ResourceClass
             );

  HLSLResourceClassAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::hlsl::ResourceClass getResourceClass() const {
    return resourceClass;
  }

  static bool ConvertStrToResourceClass(StringRef Val, llvm::hlsl::ResourceClass &Out);
  static const char *ConvertResourceClassToStr(llvm::hlsl::ResourceClass Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLResourceClass; }
};

class CLANG_ABI HLSLSV_DispatchThreadIDAttr : public HLSLAnnotationAttr {
public:
  // Factory methods
  static HLSLSV_DispatchThreadIDAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_DispatchThreadIDAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_DispatchThreadIDAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLSV_DispatchThreadIDAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLSV_DispatchThreadIDAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLSV_DispatchThreadIDAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLSV_DispatchThreadID; }
};

class CLANG_ABI HLSLSV_GroupIDAttr : public HLSLAnnotationAttr {
public:
  // Factory methods
  static HLSLSV_GroupIDAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupIDAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupIDAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLSV_GroupIDAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLSV_GroupIDAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLSV_GroupIDAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLSV_GroupID; }
};

class CLANG_ABI HLSLSV_GroupIndexAttr : public HLSLAnnotationAttr {
public:
  // Factory methods
  static HLSLSV_GroupIndexAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupIndexAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupIndexAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLSV_GroupIndexAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLSV_GroupIndexAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLSV_GroupIndexAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLSV_GroupIndex; }
};

class CLANG_ABI HLSLSV_GroupThreadIDAttr : public HLSLAnnotationAttr {
public:
  // Factory methods
  static HLSLSV_GroupThreadIDAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupThreadIDAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HLSLSV_GroupThreadIDAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static HLSLSV_GroupThreadIDAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  HLSLSV_GroupThreadIDAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HLSLSV_GroupThreadIDAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLSV_GroupThreadID; }
};

class CLANG_ABI HLSLShaderAttr : public InheritableAttr {
private:
  llvm::Triple::EnvironmentType type;

public:
  // Factory methods
  static HLSLShaderAttr *CreateImplicit(ASTContext &Ctx, llvm::Triple::EnvironmentType Type, const AttributeCommonInfo &CommonInfo);
  static HLSLShaderAttr *Create(ASTContext &Ctx, llvm::Triple::EnvironmentType Type, const AttributeCommonInfo &CommonInfo);
  static HLSLShaderAttr *CreateImplicit(ASTContext &Ctx, llvm::Triple::EnvironmentType Type, SourceRange Range = {});
  static HLSLShaderAttr *Create(ASTContext &Ctx, llvm::Triple::EnvironmentType Type, SourceRange Range = {});

  // Constructors
  HLSLShaderAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::Triple::EnvironmentType Type
             );

  HLSLShaderAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::Triple::EnvironmentType getType() const {
    return type;
  }

  static bool ConvertStrToEnvironmentType(StringRef Val, llvm::Triple::EnvironmentType &Out);
  static const char *ConvertEnvironmentTypeToStr(llvm::Triple::EnvironmentType Val);

  static bool isValidShaderType(llvm::Triple::EnvironmentType ShaderType) {
    return ShaderType >= llvm::Triple::Pixel && ShaderType <= llvm::Triple::Amplification;
  }


  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLShader; }
};

class CLANG_ABI HLSLWaveSizeAttr : public InheritableAttr {
int min;

int max;

int preferred;

public:
  // Factory methods
  static HLSLWaveSizeAttr *CreateImplicit(ASTContext &Ctx, int Min, int Max, int Preferred, const AttributeCommonInfo &CommonInfo);
  static HLSLWaveSizeAttr *Create(ASTContext &Ctx, int Min, int Max, int Preferred, const AttributeCommonInfo &CommonInfo);
  static HLSLWaveSizeAttr *CreateImplicit(ASTContext &Ctx, int Min, int Max, int Preferred, SourceRange Range = {});
  static HLSLWaveSizeAttr *Create(ASTContext &Ctx, int Min, int Max, int Preferred, SourceRange Range = {});

  // Constructors
  HLSLWaveSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Min
              , int Max
              , int Preferred
             );
  HLSLWaveSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Min
             );

  HLSLWaveSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getMin() const {
    return min;
  }

  int getMax() const {
    return max;
  }

  static const int DefaultMax = 0;

  int getPreferred() const {
    return preferred;
  }

  static const int DefaultPreferred = 0;


    private:
      int SpelledArgsCount = 0;

    public:
      void setSpelledArgsCount(int C) { SpelledArgsCount = C; }
      int getSpelledArgsCount() const { return SpelledArgsCount; }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::HLSLWaveSize; }
};

class CLANG_ABI HotAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_hot = 0,
    CXX11_gnu_hot = 1,
    C23_gnu_hot = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HotAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HotAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HotAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_hot);
  static HotAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_hot);

  // Constructors
  HotAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HotAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Hot; }
};

class CLANG_ABI HybridPatchableAttr : public InheritableAttr {
public:
  enum Spelling {
    Declspec_hybrid_patchable = 0,
    GNU_hybrid_patchable = 1,
    CXX11_clang_hybrid_patchable = 2,
    C23_clang_hybrid_patchable = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static HybridPatchableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HybridPatchableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static HybridPatchableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_hybrid_patchable);
  static HybridPatchableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_hybrid_patchable);

  // Constructors
  HybridPatchableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  HybridPatchableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::HybridPatchable; }
};

class CLANG_ABI IBActionAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ibaction = 0,
    CXX11_clang_ibaction = 1,
    C23_clang_ibaction = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static IBActionAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IBActionAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IBActionAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ibaction);
  static IBActionAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ibaction);

  // Constructors
  IBActionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  IBActionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::IBAction; }
};

class CLANG_ABI IBOutletAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_iboutlet = 0,
    CXX11_clang_iboutlet = 1,
    C23_clang_iboutlet = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static IBOutletAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IBOutletAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IBOutletAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_iboutlet);
  static IBOutletAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_iboutlet);

  // Constructors
  IBOutletAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  IBOutletAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::IBOutlet; }
};

class CLANG_ABI IBOutletCollectionAttr : public InheritableAttr {
TypeSourceInfo * interface_;

public:
  enum Spelling {
    GNU_iboutletcollection = 0,
    CXX11_clang_iboutletcollection = 1,
    C23_clang_iboutletcollection = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static IBOutletCollectionAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Interface, const AttributeCommonInfo &CommonInfo);
  static IBOutletCollectionAttr *Create(ASTContext &Ctx, TypeSourceInfo * Interface, const AttributeCommonInfo &CommonInfo);
  static IBOutletCollectionAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Interface, SourceRange Range = {}, Spelling S = GNU_iboutletcollection);
  static IBOutletCollectionAttr *Create(ASTContext &Ctx, TypeSourceInfo * Interface, SourceRange Range = {}, Spelling S = GNU_iboutletcollection);

  // Constructors
  IBOutletCollectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * Interface
             );
  IBOutletCollectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  IBOutletCollectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getInterface() const {
    return interface_->getType();
  }  TypeSourceInfo * getInterfaceLoc() const {
    return interface_;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::IBOutletCollection; }
};

class CLANG_ABI IFuncAttr : public Attr {
unsigned resolverLength;
char *resolver;

public:
  enum Spelling {
    GNU_ifunc = 0,
    CXX11_gnu_ifunc = 1,
    C23_gnu_ifunc = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static IFuncAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Resolver, const AttributeCommonInfo &CommonInfo);
  static IFuncAttr *Create(ASTContext &Ctx, llvm::StringRef Resolver, const AttributeCommonInfo &CommonInfo);
  static IFuncAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Resolver, SourceRange Range = {}, Spelling S = GNU_ifunc);
  static IFuncAttr *Create(ASTContext &Ctx, llvm::StringRef Resolver, SourceRange Range = {}, Spelling S = GNU_ifunc);

  // Constructors
  IFuncAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Resolver
             );

  IFuncAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getResolver() const {
    return llvm::StringRef(resolver, resolverLength);
  }
  unsigned getResolverLength() const {
    return resolverLength;
  }
  void setResolver(ASTContext &C, llvm::StringRef S) {
    resolverLength = S.size();
    this->resolver = new (C, 1) char [resolverLength];
    if (!S.empty())
      std::memcpy(this->resolver, S.data(), resolverLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::IFunc; }
};

class CLANG_ABI InitPriorityAttr : public InheritableAttr {
unsigned priority;

public:
  enum Spelling {
    GNU_init_priority = 0,
    CXX11_gnu_init_priority = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static InitPriorityAttr *CreateImplicit(ASTContext &Ctx, unsigned Priority, const AttributeCommonInfo &CommonInfo);
  static InitPriorityAttr *Create(ASTContext &Ctx, unsigned Priority, const AttributeCommonInfo &CommonInfo);
  static InitPriorityAttr *CreateImplicit(ASTContext &Ctx, unsigned Priority, SourceRange Range = {}, Spelling S = GNU_init_priority);
  static InitPriorityAttr *Create(ASTContext &Ctx, unsigned Priority, SourceRange Range = {}, Spelling S = GNU_init_priority);

  // Constructors
  InitPriorityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Priority
             );

  InitPriorityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getPriority() const {
    return priority;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::InitPriority; }
};

class CLANG_ABI InitSegAttr : public Attr {
unsigned sectionLength;
char *section;

public:
  // Factory methods
  static InitSegAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Section, const AttributeCommonInfo &CommonInfo);
  static InitSegAttr *Create(ASTContext &Ctx, llvm::StringRef Section, const AttributeCommonInfo &CommonInfo);
  static InitSegAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Section, SourceRange Range = {});
  static InitSegAttr *Create(ASTContext &Ctx, llvm::StringRef Section, SourceRange Range = {});

  // Constructors
  InitSegAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Section
             );

  InitSegAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getSection() const {
    return llvm::StringRef(section, sectionLength);
  }
  unsigned getSectionLength() const {
    return sectionLength;
  }
  void setSection(ASTContext &C, llvm::StringRef S) {
    sectionLength = S.size();
    this->section = new (C, 1) char [sectionLength];
    if (!S.empty())
      std::memcpy(this->section, S.data(), sectionLength);
  }


  void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const {
    OS << " (" << getSection() << ')';
  }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::InitSeg; }
};

class CLANG_ABI IntelOclBiccAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_intel_ocl_bicc = 0,
    CXX11_clang_intel_ocl_bicc = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static IntelOclBiccAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IntelOclBiccAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static IntelOclBiccAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_intel_ocl_bicc);
  static IntelOclBiccAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_intel_ocl_bicc);

  // Constructors
  IntelOclBiccAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  IntelOclBiccAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::IntelOclBicc; }
};

class CLANG_ABI InternalLinkageAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_internal_linkage = 0,
    CXX11_clang_internal_linkage = 1,
    C23_clang_internal_linkage = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static InternalLinkageAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static InternalLinkageAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static InternalLinkageAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_internal_linkage);
  static InternalLinkageAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_internal_linkage);

  // Constructors
  InternalLinkageAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  InternalLinkageAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::InternalLinkage; }
};

class CLANG_ABI LTOVisibilityPublicAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_lto_visibility_public = 0,
    CXX11_clang_lto_visibility_public = 1,
    C23_clang_lto_visibility_public = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LTOVisibilityPublicAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LTOVisibilityPublicAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LTOVisibilityPublicAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_lto_visibility_public);
  static LTOVisibilityPublicAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_lto_visibility_public);

  // Constructors
  LTOVisibilityPublicAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LTOVisibilityPublicAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::LTOVisibilityPublic; }
};

class CLANG_ABI LayoutVersionAttr : public InheritableAttr {
unsigned version;

public:
  // Factory methods
  static LayoutVersionAttr *CreateImplicit(ASTContext &Ctx, unsigned Version, const AttributeCommonInfo &CommonInfo);
  static LayoutVersionAttr *Create(ASTContext &Ctx, unsigned Version, const AttributeCommonInfo &CommonInfo);
  static LayoutVersionAttr *CreateImplicit(ASTContext &Ctx, unsigned Version, SourceRange Range = {});
  static LayoutVersionAttr *Create(ASTContext &Ctx, unsigned Version, SourceRange Range = {});

  // Constructors
  LayoutVersionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Version
             );

  LayoutVersionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getVersion() const {
    return version;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::LayoutVersion; }
};

class CLANG_ABI LeafAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_leaf = 0,
    CXX11_gnu_leaf = 1,
    C23_gnu_leaf = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LeafAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LeafAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LeafAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_leaf);
  static LeafAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_leaf);

  // Constructors
  LeafAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LeafAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Leaf; }
};

class CLANG_ABI LifetimeBoundAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_lifetimebound = 0,
    CXX11_clang_lifetimebound = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LifetimeBoundAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LifetimeBoundAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_lifetimebound);
  static LifetimeBoundAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_lifetimebound);

  // Constructors
  LifetimeBoundAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LifetimeBoundAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::LifetimeBound; }
};

class CLANG_ABI LifetimeCaptureByAttr : public InheritableAttr {
  unsigned params_Size;
  int *params_;

public:
  enum Spelling {
    GNU_lifetime_capture_by = 0,
    CXX11_clang_lifetime_capture_by = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LifetimeCaptureByAttr *CreateImplicit(ASTContext &Ctx, int *Params, unsigned ParamsSize, const AttributeCommonInfo &CommonInfo);
  static LifetimeCaptureByAttr *Create(ASTContext &Ctx, int *Params, unsigned ParamsSize, const AttributeCommonInfo &CommonInfo);
  static LifetimeCaptureByAttr *CreateImplicit(ASTContext &Ctx, int *Params, unsigned ParamsSize, SourceRange Range = {}, Spelling S = GNU_lifetime_capture_by);
  static LifetimeCaptureByAttr *Create(ASTContext &Ctx, int *Params, unsigned ParamsSize, SourceRange Range = {}, Spelling S = GNU_lifetime_capture_by);

  // Constructors
  LifetimeCaptureByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int *Params, unsigned ParamsSize
             );
  LifetimeCaptureByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LifetimeCaptureByAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef int* params_iterator;
  params_iterator params_begin() const { return params_; }
  params_iterator params_end() const { return params_ + params_Size; }
  unsigned params_size() const { return params_Size; }
  llvm::iterator_range<params_iterator> params() const { return llvm::make_range(params_begin(), params_end()); }



private:
  ArrayRef<IdentifierInfo*> ArgIdents;
  ArrayRef<SourceLocation> ArgLocs;

public:
  static constexpr int THIS = 0;
  static constexpr int INVALID = -1;
  static constexpr int UNKNOWN = -2;
  static constexpr int GLOBAL = -3;

  void setArgs(ArrayRef<IdentifierInfo*> Idents, ArrayRef<SourceLocation> Locs) {
    assert(Idents.size() == params_Size);
    assert(Locs.size() == params_Size);
    ArgIdents = Idents;
    ArgLocs = Locs;
  }
  auto getArgIdents() const { return ArgIdents; }
  auto getArgLocs() const { return ArgLocs; }
  void setParamIdx(size_t Idx, int Val) {
    assert(Idx < params_Size);
    params_[Idx] = Val;
  }


  static bool classof(const Attr *A) { return A->getKind() == attr::LifetimeCaptureBy; }
};

class CLANG_ABI LikelyAttr : public StmtAttr {
public:
  enum Spelling {
    CXX11_likely = 0,
    C23_clang_likely = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LikelyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LikelyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LikelyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_likely);
  static LikelyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_likely);

  // Constructors
  LikelyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LikelyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Likely; }
};

class CLANG_ABI LoaderUninitializedAttr : public Attr {
public:
  enum Spelling {
    GNU_loader_uninitialized = 0,
    CXX11_clang_loader_uninitialized = 1,
    C23_clang_loader_uninitialized = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LoaderUninitializedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LoaderUninitializedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static LoaderUninitializedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_loader_uninitialized);
  static LoaderUninitializedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_loader_uninitialized);

  // Constructors
  LoaderUninitializedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LoaderUninitializedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::LoaderUninitialized; }
};

class CLANG_ABI LockReturnedAttr : public InheritableAttr {
Expr * arg;

public:
  // Factory methods
  static LockReturnedAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static LockReturnedAttr *Create(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static LockReturnedAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});
  static LockReturnedAttr *Create(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});

  // Constructors
  LockReturnedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Arg
             );

  LockReturnedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getArg() const {
    return arg;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::LockReturned; }
};

class CLANG_ABI LocksExcludedAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  // Factory methods
  static LocksExcludedAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static LocksExcludedAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static LocksExcludedAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});
  static LocksExcludedAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {});

  // Constructors
  LocksExcludedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  LocksExcludedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  LocksExcludedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::LocksExcluded; }
};

class CLANG_ABI LoopHintAttr : public Attr {
public:
  enum OptionType {
    Vectorize,
    VectorizeWidth,
    Interleave,
    InterleaveCount,
    Unroll,
    UnrollCount,
    UnrollAndJam,
    UnrollAndJamCount,
    PipelineDisabled,
    PipelineInitiationInterval,
    Distribute,
    VectorizePredicate
  };
private:
  LoopHintAttr::OptionType option;

public:
  enum LoopHintState {
    Enable,
    Disable,
    Numeric,
    FixedWidth,
    ScalableWidth,
    AssumeSafety,
    Full
  };
private:
  LoopHintAttr::LoopHintState state;

Expr * value;

public:
  enum Spelling {
    Pragma_clang_loop = 0,
    Pragma_unroll = 1,
    Pragma_nounroll = 2,
    Pragma_unroll_and_jam = 3,
    Pragma_nounroll_and_jam = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static LoopHintAttr *CreateImplicit(ASTContext &Ctx, LoopHintAttr::OptionType Option, LoopHintAttr::LoopHintState State, Expr * Value, const AttributeCommonInfo &CommonInfo);
  static LoopHintAttr *Create(ASTContext &Ctx, LoopHintAttr::OptionType Option, LoopHintAttr::LoopHintState State, Expr * Value, const AttributeCommonInfo &CommonInfo);
  static LoopHintAttr *CreateImplicit(ASTContext &Ctx, LoopHintAttr::OptionType Option, LoopHintAttr::LoopHintState State, Expr * Value, SourceRange Range = {}, Spelling S = Pragma_clang_loop);
  static LoopHintAttr *Create(ASTContext &Ctx, LoopHintAttr::OptionType Option, LoopHintAttr::LoopHintState State, Expr * Value, SourceRange Range = {}, Spelling S = Pragma_clang_loop);

  // Constructors
  LoopHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , LoopHintAttr::OptionType Option
              , LoopHintAttr::LoopHintState State
              , Expr * Value
             );

  LoopHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  LoopHintAttr::OptionType getOption() const {
    return option;
  }

  static bool ConvertStrToOptionType(StringRef Val, LoopHintAttr::OptionType &Out);
  static const char *ConvertOptionTypeToStr(LoopHintAttr::OptionType Val);
  LoopHintAttr::LoopHintState getState() const {
    return state;
  }

  static bool ConvertStrToLoopHintState(StringRef Val, LoopHintAttr::LoopHintState &Out);
  static const char *ConvertLoopHintStateToStr(LoopHintAttr::LoopHintState Val);
  Expr * getValue() const {
    return value;
  }


  static const char *getOptionName(int Option) {
    switch(Option) {
    case Vectorize: return "vectorize";
    case VectorizeWidth: return "vectorize_width";
    case Interleave: return "interleave";
    case InterleaveCount: return "interleave_count";
    case Unroll: return "unroll";
    case UnrollCount: return "unroll_count";
    case UnrollAndJam: return "unroll_and_jam";
    case UnrollAndJamCount: return "unroll_and_jam_count";
    case PipelineDisabled: return "pipeline";
    case PipelineInitiationInterval: return "pipeline_initiation_interval";
    case Distribute: return "distribute";
    case VectorizePredicate: return "vectorize_predicate";
    }
    llvm_unreachable("Unhandled LoopHint option.");
  }

  void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const;

  // Return a string containing the loop hint argument including the
  // enclosing parentheses.
  std::string getValueString(const PrintingPolicy &Policy) const;

  // Return a string suitable for identifying this attribute in diagnostics.
  std::string getDiagnosticName(const PrintingPolicy &Policy) const;
  

  static bool classof(const Attr *A) { return A->getKind() == attr::LoopHint; }
};

class CLANG_ABI M68kInterruptAttr : public InheritableAttr {
unsigned number;

public:
  // Factory methods
  static M68kInterruptAttr *CreateImplicit(ASTContext &Ctx, unsigned Number, const AttributeCommonInfo &CommonInfo);
  static M68kInterruptAttr *Create(ASTContext &Ctx, unsigned Number, const AttributeCommonInfo &CommonInfo);
  static M68kInterruptAttr *CreateImplicit(ASTContext &Ctx, unsigned Number, SourceRange Range = {});
  static M68kInterruptAttr *Create(ASTContext &Ctx, unsigned Number, SourceRange Range = {});

  // Constructors
  M68kInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Number
             );

  M68kInterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getNumber() const {
    return number;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::M68kInterrupt; }
};

class CLANG_ABI M68kRTDAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_m68k_rtd = 0,
    CXX11_clang_m68k_rtd = 1,
    C23_clang_m68k_rtd = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static M68kRTDAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static M68kRTDAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static M68kRTDAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_m68k_rtd);
  static M68kRTDAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_m68k_rtd);

  // Constructors
  M68kRTDAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  M68kRTDAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::M68kRTD; }
};

class CLANG_ABI MIGServerRoutineAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_mig_server_routine = 0,
    CXX11_clang_mig_server_routine = 1,
    C23_clang_mig_server_routine = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MIGServerRoutineAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MIGServerRoutineAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MIGServerRoutineAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_mig_server_routine);
  static MIGServerRoutineAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_mig_server_routine);

  // Constructors
  MIGServerRoutineAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MIGServerRoutineAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MIGServerRoutine; }
};

class CLANG_ABI MSABIAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ms_abi = 0,
    CXX11_gnu_ms_abi = 1,
    C23_gnu_ms_abi = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MSABIAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSABIAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSABIAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ms_abi);
  static MSABIAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ms_abi);

  // Constructors
  MSABIAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSABIAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MSABI; }
};

class CLANG_ABI MSAllocatorAttr : public InheritableAttr {
public:
  // Factory methods
  static MSAllocatorAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSAllocatorAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSAllocatorAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static MSAllocatorAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  MSAllocatorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSAllocatorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MSAllocator; }
};

class CLANG_ABI MSConstexprAttr : public InheritableAttr {
public:
  // Factory methods
  static MSConstexprAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSConstexprAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSConstexprAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static MSConstexprAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  MSConstexprAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSConstexprAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MSConstexpr; }
};

class CLANG_ABI MSInheritanceAttr : public InheritableAttr {
bool bestCase;

public:
  enum Spelling {
    Keyword_single_inheritance = 0,
    Keyword_multiple_inheritance = 1,
    Keyword_virtual_inheritance = 2,
    Keyword_unspecified_inheritance = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MSInheritanceAttr *CreateImplicit(ASTContext &Ctx, bool BestCase, const AttributeCommonInfo &CommonInfo);
  static MSInheritanceAttr *Create(ASTContext &Ctx, bool BestCase, const AttributeCommonInfo &CommonInfo);
  static MSInheritanceAttr *CreateImplicit(ASTContext &Ctx, bool BestCase, SourceRange Range = {}, Spelling S = Keyword_single_inheritance);
  static MSInheritanceAttr *Create(ASTContext &Ctx, bool BestCase, SourceRange Range = {}, Spelling S = Keyword_single_inheritance);
  static MSInheritanceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSInheritanceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSInheritanceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_single_inheritance);
  static MSInheritanceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_single_inheritance);

  // Constructors
  MSInheritanceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , bool BestCase
             );
  MSInheritanceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSInheritanceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool getBestCase() const {
    return bestCase;
  }

  static const bool DefaultBestCase = true;


  MSInheritanceModel getInheritanceModel() const {
    // The spelling enum should agree with MSInheritanceModel.
    return MSInheritanceModel(getSemanticSpelling());
  }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::MSInheritance; }
};

class CLANG_ABI MSNoVTableAttr : public InheritableAttr {
public:
  // Factory methods
  static MSNoVTableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSNoVTableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSNoVTableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static MSNoVTableAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  MSNoVTableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSNoVTableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MSNoVTable; }
};

class CLANG_ABI MSP430InterruptAttr : public InheritableAttr {
unsigned number;

public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MSP430InterruptAttr *CreateImplicit(ASTContext &Ctx, unsigned Number, const AttributeCommonInfo &CommonInfo);
  static MSP430InterruptAttr *Create(ASTContext &Ctx, unsigned Number, const AttributeCommonInfo &CommonInfo);
  static MSP430InterruptAttr *CreateImplicit(ASTContext &Ctx, unsigned Number, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static MSP430InterruptAttr *Create(ASTContext &Ctx, unsigned Number, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  MSP430InterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Number
             );

  MSP430InterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getNumber() const {
    return number;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::MSP430Interrupt; }
};

class CLANG_ABI MSStructAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ms_struct = 0,
    CXX11_gnu_ms_struct = 1,
    C23_gnu_ms_struct = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MSStructAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSStructAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MSStructAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ms_struct);
  static MSStructAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ms_struct);

  // Constructors
  MSStructAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MSStructAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MSStruct; }
};

class CLANG_ABI MSVtorDispAttr : public InheritableAttr {
unsigned vdm;

public:
  // Factory methods
  static MSVtorDispAttr *CreateImplicit(ASTContext &Ctx, unsigned Vdm, const AttributeCommonInfo &CommonInfo);
  static MSVtorDispAttr *Create(ASTContext &Ctx, unsigned Vdm, const AttributeCommonInfo &CommonInfo);
  static MSVtorDispAttr *CreateImplicit(ASTContext &Ctx, unsigned Vdm, SourceRange Range = {});
  static MSVtorDispAttr *Create(ASTContext &Ctx, unsigned Vdm, SourceRange Range = {});

  // Constructors
  MSVtorDispAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Vdm
             );

  MSVtorDispAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getVdm() const {
    return vdm;
  }


  MSVtorDispMode getVtorDispMode() const { return MSVtorDispMode(vdm); }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::MSVtorDisp; }
};

class CLANG_ABI MaxFieldAlignmentAttr : public InheritableAttr {
unsigned alignment;

public:
  // Factory methods
  static MaxFieldAlignmentAttr *CreateImplicit(ASTContext &Ctx, unsigned Alignment, const AttributeCommonInfo &CommonInfo);
  static MaxFieldAlignmentAttr *Create(ASTContext &Ctx, unsigned Alignment, const AttributeCommonInfo &CommonInfo);
  static MaxFieldAlignmentAttr *CreateImplicit(ASTContext &Ctx, unsigned Alignment, SourceRange Range = {});
  static MaxFieldAlignmentAttr *Create(ASTContext &Ctx, unsigned Alignment, SourceRange Range = {});

  // Constructors
  MaxFieldAlignmentAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Alignment
             );

  MaxFieldAlignmentAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getAlignment() const {
    return alignment;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::MaxFieldAlignment; }
};

class CLANG_ABI MayAliasAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_may_alias = 0,
    CXX11_gnu_may_alias = 1,
    C23_gnu_may_alias = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MayAliasAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MayAliasAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MayAliasAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_may_alias);
  static MayAliasAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_may_alias);

  // Constructors
  MayAliasAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MayAliasAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MayAlias; }
};

class CLANG_ABI MaybeUndefAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_maybe_undef = 0,
    CXX11_clang_maybe_undef = 1,
    C23_clang_maybe_undef = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MaybeUndefAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MaybeUndefAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MaybeUndefAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_maybe_undef);
  static MaybeUndefAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_maybe_undef);

  // Constructors
  MaybeUndefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MaybeUndefAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MaybeUndef; }
};

class CLANG_ABI MicroMipsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_micromips = 0,
    CXX11_gnu_micromips = 1,
    C23_gnu_micromips = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MicroMipsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MicroMipsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MicroMipsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_micromips);
  static MicroMipsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_micromips);

  // Constructors
  MicroMipsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MicroMipsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MicroMips; }
};

class CLANG_ABI MinSizeAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_minsize = 0,
    CXX11_clang_minsize = 1,
    C23_clang_minsize = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MinSizeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MinSizeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MinSizeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_minsize);
  static MinSizeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_minsize);

  // Constructors
  MinSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MinSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MinSize; }
};

class CLANG_ABI MinVectorWidthAttr : public InheritableAttr {
unsigned vectorWidth;

public:
  enum Spelling {
    GNU_min_vector_width = 0,
    CXX11_clang_min_vector_width = 1,
    C23_clang_min_vector_width = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MinVectorWidthAttr *CreateImplicit(ASTContext &Ctx, unsigned VectorWidth, const AttributeCommonInfo &CommonInfo);
  static MinVectorWidthAttr *Create(ASTContext &Ctx, unsigned VectorWidth, const AttributeCommonInfo &CommonInfo);
  static MinVectorWidthAttr *CreateImplicit(ASTContext &Ctx, unsigned VectorWidth, SourceRange Range = {}, Spelling S = GNU_min_vector_width);
  static MinVectorWidthAttr *Create(ASTContext &Ctx, unsigned VectorWidth, SourceRange Range = {}, Spelling S = GNU_min_vector_width);

  // Constructors
  MinVectorWidthAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned VectorWidth
             );

  MinVectorWidthAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getVectorWidth() const {
    return vectorWidth;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::MinVectorWidth; }
};

class CLANG_ABI Mips16Attr : public InheritableAttr {
public:
  enum Spelling {
    GNU_mips16 = 0,
    CXX11_gnu_mips16 = 1,
    C23_gnu_mips16 = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static Mips16Attr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Mips16Attr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Mips16Attr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_mips16);
  static Mips16Attr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_mips16);

  // Constructors
  Mips16Attr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  Mips16Attr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Mips16; }
};

class CLANG_ABI MipsInterruptAttr : public InheritableAttr {
public:
  enum InterruptType {
    sw0,
    sw1,
    hw0,
    hw1,
    hw2,
    hw3,
    hw4,
    hw5,
    eic
  };
private:
  MipsInterruptAttr::InterruptType interrupt;

public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MipsInterruptAttr *CreateImplicit(ASTContext &Ctx, MipsInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static MipsInterruptAttr *Create(ASTContext &Ctx, MipsInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static MipsInterruptAttr *CreateImplicit(ASTContext &Ctx, MipsInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static MipsInterruptAttr *Create(ASTContext &Ctx, MipsInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  MipsInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , MipsInterruptAttr::InterruptType Interrupt
             );

  MipsInterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  MipsInterruptAttr::InterruptType getInterrupt() const {
    return interrupt;
  }

  static bool ConvertStrToInterruptType(StringRef Val, MipsInterruptAttr::InterruptType &Out);
  static const char *ConvertInterruptTypeToStr(MipsInterruptAttr::InterruptType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::MipsInterrupt; }
};

class CLANG_ABI MipsLongCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_long_call = 0,
    CXX11_gnu_long_call = 1,
    C23_gnu_long_call = 2,
    GNU_far = 3,
    CXX11_gnu_far = 4,
    C23_gnu_far = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MipsLongCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MipsLongCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MipsLongCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_long_call);
  static MipsLongCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_long_call);

  // Constructors
  MipsLongCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MipsLongCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MipsLongCall; }
};

class CLANG_ABI MipsShortCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_short_call = 0,
    CXX11_gnu_short_call = 1,
    C23_gnu_short_call = 2,
    GNU_near = 3,
    CXX11_gnu_near = 4,
    C23_gnu_near = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MipsShortCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MipsShortCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MipsShortCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_short_call);
  static MipsShortCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_short_call);

  // Constructors
  MipsShortCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MipsShortCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MipsShortCall; }
};

class CLANG_ABI ModeAttr : public Attr {
IdentifierInfo * mode;

public:
  enum Spelling {
    GNU_mode = 0,
    CXX11_gnu_mode = 1,
    C23_gnu_mode = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ModeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Mode, const AttributeCommonInfo &CommonInfo);
  static ModeAttr *Create(ASTContext &Ctx, IdentifierInfo * Mode, const AttributeCommonInfo &CommonInfo);
  static ModeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Mode, SourceRange Range = {}, Spelling S = GNU_mode);
  static ModeAttr *Create(ASTContext &Ctx, IdentifierInfo * Mode, SourceRange Range = {}, Spelling S = GNU_mode);

  // Constructors
  ModeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Mode
             );

  ModeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getMode() const {
    return mode;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Mode; }
};

class CLANG_ABI MustTailAttr : public StmtAttr {
public:
  enum Spelling {
    GNU_musttail = 0,
    CXX11_clang_musttail = 1,
    C23_clang_musttail = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static MustTailAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MustTailAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static MustTailAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_musttail);
  static MustTailAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_musttail);

  // Constructors
  MustTailAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  MustTailAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::MustTail; }
};

class CLANG_ABI NSConsumedAttr : public InheritableParamAttr {
public:
  enum Spelling {
    GNU_ns_consumed = 0,
    CXX11_clang_ns_consumed = 1,
    C23_clang_ns_consumed = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NSConsumedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSConsumedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSConsumedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_consumed);
  static NSConsumedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_consumed);

  // Constructors
  NSConsumedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NSConsumedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NSConsumed; }
};

class CLANG_ABI NSConsumesSelfAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ns_consumes_self = 0,
    CXX11_clang_ns_consumes_self = 1,
    C23_clang_ns_consumes_self = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NSConsumesSelfAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSConsumesSelfAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSConsumesSelfAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_consumes_self);
  static NSConsumesSelfAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_consumes_self);

  // Constructors
  NSConsumesSelfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NSConsumesSelfAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NSConsumesSelf; }
};

class CLANG_ABI NSErrorDomainAttr : public InheritableAttr {
IdentifierInfo * errorDomain;

public:
  // Factory methods
  static NSErrorDomainAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ErrorDomain, const AttributeCommonInfo &CommonInfo);
  static NSErrorDomainAttr *Create(ASTContext &Ctx, IdentifierInfo * ErrorDomain, const AttributeCommonInfo &CommonInfo);
  static NSErrorDomainAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ErrorDomain, SourceRange Range = {});
  static NSErrorDomainAttr *Create(ASTContext &Ctx, IdentifierInfo * ErrorDomain, SourceRange Range = {});

  // Constructors
  NSErrorDomainAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * ErrorDomain
             );

  NSErrorDomainAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getErrorDomain() const {
    return errorDomain;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::NSErrorDomain; }
};

class CLANG_ABI NSReturnsAutoreleasedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ns_returns_autoreleased = 0,
    CXX11_clang_ns_returns_autoreleased = 1,
    C23_clang_ns_returns_autoreleased = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NSReturnsAutoreleasedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsAutoreleasedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsAutoreleasedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_autoreleased);
  static NSReturnsAutoreleasedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_autoreleased);

  // Constructors
  NSReturnsAutoreleasedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NSReturnsAutoreleasedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NSReturnsAutoreleased; }
};

class CLANG_ABI NSReturnsNotRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ns_returns_not_retained = 0,
    CXX11_clang_ns_returns_not_retained = 1,
    C23_clang_ns_returns_not_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NSReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsNotRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_not_retained);
  static NSReturnsNotRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_not_retained);

  // Constructors
  NSReturnsNotRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NSReturnsNotRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NSReturnsNotRetained; }
};

class CLANG_ABI NSReturnsRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_ns_returns_retained = 0,
    CXX11_clang_ns_returns_retained = 1,
    C23_clang_ns_returns_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NSReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NSReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_retained);
  static NSReturnsRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_ns_returns_retained);

  // Constructors
  NSReturnsRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NSReturnsRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NSReturnsRetained; }
};

class CLANG_ABI NVPTXKernelAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nvptx_kernel = 0,
    CXX11_clang_nvptx_kernel = 1,
    C23_clang_nvptx_kernel = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NVPTXKernelAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NVPTXKernelAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NVPTXKernelAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nvptx_kernel);
  static NVPTXKernelAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nvptx_kernel);

  // Constructors
  NVPTXKernelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NVPTXKernelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NVPTXKernel; }
};

class CLANG_ABI NakedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_naked = 0,
    CXX11_gnu_naked = 1,
    C23_gnu_naked = 2,
    Declspec_naked = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NakedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NakedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NakedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_naked);
  static NakedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_naked);

  // Constructors
  NakedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NakedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Naked; }
};

class CLANG_ABI NoAliasAttr : public InheritableAttr {
public:
  // Factory methods
  static NoAliasAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoAliasAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoAliasAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static NoAliasAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  NoAliasAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoAliasAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoAlias; }
};

class CLANG_ABI NoBuiltinAttr : public Attr {
  unsigned builtinNames_Size;
  StringRef *builtinNames_;

public:
  enum Spelling {
    GNU_no_builtin = 0,
    CXX11_clang_no_builtin = 1,
    C23_clang_no_builtin = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoBuiltinAttr *CreateImplicit(ASTContext &Ctx, StringRef *BuiltinNames, unsigned BuiltinNamesSize, const AttributeCommonInfo &CommonInfo);
  static NoBuiltinAttr *Create(ASTContext &Ctx, StringRef *BuiltinNames, unsigned BuiltinNamesSize, const AttributeCommonInfo &CommonInfo);
  static NoBuiltinAttr *CreateImplicit(ASTContext &Ctx, StringRef *BuiltinNames, unsigned BuiltinNamesSize, SourceRange Range = {}, Spelling S = GNU_no_builtin);
  static NoBuiltinAttr *Create(ASTContext &Ctx, StringRef *BuiltinNames, unsigned BuiltinNamesSize, SourceRange Range = {}, Spelling S = GNU_no_builtin);

  // Constructors
  NoBuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *BuiltinNames, unsigned BuiltinNamesSize
             );
  NoBuiltinAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoBuiltinAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* builtinNames_iterator;
  builtinNames_iterator builtinNames_begin() const { return builtinNames_; }
  builtinNames_iterator builtinNames_end() const { return builtinNames_ + builtinNames_Size; }
  unsigned builtinNames_size() const { return builtinNames_Size; }
  llvm::iterator_range<builtinNames_iterator> builtinNames() const { return llvm::make_range(builtinNames_begin(), builtinNames_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::NoBuiltin; }
};

class CLANG_ABI NoCommonAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nocommon = 0,
    CXX11_gnu_nocommon = 1,
    C23_gnu_nocommon = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoCommonAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoCommonAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoCommonAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nocommon);
  static NoCommonAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nocommon);

  // Constructors
  NoCommonAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoCommonAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoCommon; }
};

class CLANG_ABI NoConvergentAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_noconvergent = 0,
    CXX11_clang_noconvergent = 1,
    C23_clang_noconvergent = 2,
    Declspec_noconvergent = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoConvergentAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoConvergentAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoConvergentAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noconvergent);
  static NoConvergentAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noconvergent);

  // Constructors
  NoConvergentAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoConvergentAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoConvergent; }
};

class CLANG_ABI NoDebugAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nodebug = 0,
    CXX11_gnu_nodebug = 1,
    C23_gnu_nodebug = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoDebugAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDebugAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDebugAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nodebug);
  static NoDebugAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nodebug);

  // Constructors
  NoDebugAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoDebugAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoDebug; }
};

class CLANG_ABI NoDerefAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_noderef = 0,
    CXX11_clang_noderef = 1,
    C23_clang_noderef = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoDerefAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDerefAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDerefAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noderef);
  static NoDerefAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noderef);

  // Constructors
  NoDerefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoDerefAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoDeref; }
};

class CLANG_ABI NoDestroyAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_destroy = 0,
    CXX11_clang_no_destroy = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoDestroyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDestroyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDestroyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_destroy);
  static NoDestroyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_destroy);

  // Constructors
  NoDestroyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoDestroyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoDestroy; }
};

class CLANG_ABI NoDuplicateAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_noduplicate = 0,
    CXX11_clang_noduplicate = 1,
    C23_clang_noduplicate = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoDuplicateAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDuplicateAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoDuplicateAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noduplicate);
  static NoDuplicateAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noduplicate);

  // Constructors
  NoDuplicateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoDuplicateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoDuplicate; }
};

class CLANG_ABI NoEscapeAttr : public Attr {
public:
  enum Spelling {
    GNU_noescape = 0,
    CXX11_clang_noescape = 1,
    C23_clang_noescape = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoEscapeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoEscapeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoEscapeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noescape);
  static NoEscapeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noescape);

  // Constructors
  NoEscapeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoEscapeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoEscape; }
};

class CLANG_ABI NoInlineAttr : public DeclOrStmtAttr {
public:
  enum Spelling {
    Keyword_noinline = 0,
    GNU_noinline = 1,
    CXX11_gnu_noinline = 2,
    C23_gnu_noinline = 3,
    CXX11_clang_noinline = 4,
    C23_clang_noinline = 5,
    CXX11_msvc_noinline = 6,
    C23_msvc_noinline = 7,
    Declspec_noinline = 8,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoInlineAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoInlineAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoInlineAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_noinline);
  static NoInlineAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_noinline);

  // Constructors
  NoInlineAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoInlineAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  bool isStmtNoInline() const { return getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5 ||
    getAttributeSpellingListIndex() == 6 ||
    getAttributeSpellingListIndex() == 7; }


  static bool classof(const Attr *A) { return A->getKind() == attr::NoInline; }
};

class CLANG_ABI NoInstrumentFunctionAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_instrument_function = 0,
    CXX11_gnu_no_instrument_function = 1,
    C23_gnu_no_instrument_function = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoInstrumentFunctionAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoInstrumentFunctionAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoInstrumentFunctionAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_instrument_function);
  static NoInstrumentFunctionAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_instrument_function);

  // Constructors
  NoInstrumentFunctionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoInstrumentFunctionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoInstrumentFunction; }
};

class CLANG_ABI NoMergeAttr : public DeclOrStmtAttr {
public:
  enum Spelling {
    GNU_nomerge = 0,
    CXX11_clang_nomerge = 1,
    C23_clang_nomerge = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoMergeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMergeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMergeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomerge);
  static NoMergeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomerge);

  // Constructors
  NoMergeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoMergeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoMerge; }
};

class CLANG_ABI NoMicroMipsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nomicromips = 0,
    CXX11_gnu_nomicromips = 1,
    C23_gnu_nomicromips = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoMicroMipsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMicroMipsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMicroMipsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomicromips);
  static NoMicroMipsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomicromips);

  // Constructors
  NoMicroMipsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoMicroMipsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoMicroMips; }
};

class CLANG_ABI NoMips16Attr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nomips16 = 0,
    CXX11_gnu_nomips16 = 1,
    C23_gnu_nomips16 = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoMips16Attr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMips16Attr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoMips16Attr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomips16);
  static NoMips16Attr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nomips16);

  // Constructors
  NoMips16Attr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoMips16Attr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoMips16; }
};

class CLANG_ABI NoProfileFunctionAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_profile_instrument_function = 0,
    CXX11_gnu_no_profile_instrument_function = 1,
    C23_gnu_no_profile_instrument_function = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoProfileFunctionAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoProfileFunctionAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoProfileFunctionAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_profile_instrument_function);
  static NoProfileFunctionAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_profile_instrument_function);

  // Constructors
  NoProfileFunctionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoProfileFunctionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoProfileFunction; }
};

class CLANG_ABI NoRandomizeLayoutAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_randomize_layout = 0,
    CXX11_gnu_no_randomize_layout = 1,
    C23_gnu_no_randomize_layout = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoRandomizeLayoutAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoRandomizeLayoutAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoRandomizeLayoutAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_randomize_layout);
  static NoRandomizeLayoutAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_randomize_layout);

  // Constructors
  NoRandomizeLayoutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoRandomizeLayoutAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoRandomizeLayout; }
};

class CLANG_ABI NoReturnAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_noreturn = 0,
    CXX11_gnu_noreturn = 1,
    C23_gnu_noreturn = 2,
    Declspec_noreturn = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoReturnAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoReturnAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoReturnAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noreturn);
  static NoReturnAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_noreturn);

  // Constructors
  NoReturnAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoReturnAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoReturn; }
};

class CLANG_ABI NoSanitizeAttr : public InheritableAttr {
  unsigned sanitizers_Size;
  StringRef *sanitizers_;

public:
  enum Spelling {
    GNU_no_sanitize = 0,
    CXX11_clang_no_sanitize = 1,
    CXX11_gnu_no_sanitize = 2,
    C23_clang_no_sanitize = 3,
    C23_gnu_no_sanitize = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoSanitizeAttr *CreateImplicit(ASTContext &Ctx, StringRef *Sanitizers, unsigned SanitizersSize, const AttributeCommonInfo &CommonInfo);
  static NoSanitizeAttr *Create(ASTContext &Ctx, StringRef *Sanitizers, unsigned SanitizersSize, const AttributeCommonInfo &CommonInfo);
  static NoSanitizeAttr *CreateImplicit(ASTContext &Ctx, StringRef *Sanitizers, unsigned SanitizersSize, SourceRange Range = {}, Spelling S = GNU_no_sanitize);
  static NoSanitizeAttr *Create(ASTContext &Ctx, StringRef *Sanitizers, unsigned SanitizersSize, SourceRange Range = {}, Spelling S = GNU_no_sanitize);

  // Constructors
  NoSanitizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *Sanitizers, unsigned SanitizersSize
             );
  NoSanitizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoSanitizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* sanitizers_iterator;
  sanitizers_iterator sanitizers_begin() const { return sanitizers_; }
  sanitizers_iterator sanitizers_end() const { return sanitizers_ + sanitizers_Size; }
  unsigned sanitizers_size() const { return sanitizers_Size; }
  llvm::iterator_range<sanitizers_iterator> sanitizers() const { return llvm::make_range(sanitizers_begin(), sanitizers_end()); }



    SanitizerMask getMask() const {
      SanitizerMask Mask;
      for (auto SanitizerName : sanitizers()) {
        SanitizerMask ParsedMask =
            parseSanitizerValue(SanitizerName, /*AllowGroups=*/true);
        Mask |= expandSanitizerGroups(ParsedMask);
      }
      return Mask;
    }

    bool hasCoverage() const {
      return llvm::is_contained(sanitizers(), "coverage");
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::NoSanitize; }
};

class CLANG_ABI NoSpecializationsAttr : public InheritableAttr {
unsigned messageLength;
char *message;

public:
  enum Spelling {
    GNU_no_specializations = 0,
    CXX11_clang_no_specializations = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoSpecializationsAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static NoSpecializationsAttr *Create(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static NoSpecializationsAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = GNU_no_specializations);
  static NoSpecializationsAttr *Create(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = GNU_no_specializations);

  // Constructors
  NoSpecializationsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Message
             );
  NoSpecializationsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoSpecializationsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::NoSpecializations; }
};

class CLANG_ABI NoSpeculativeLoadHardeningAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_speculative_load_hardening = 0,
    CXX11_clang_no_speculative_load_hardening = 1,
    C23_clang_no_speculative_load_hardening = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoSpeculativeLoadHardeningAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoSpeculativeLoadHardeningAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoSpeculativeLoadHardeningAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_speculative_load_hardening);
  static NoSpeculativeLoadHardeningAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_speculative_load_hardening);

  // Constructors
  NoSpeculativeLoadHardeningAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoSpeculativeLoadHardeningAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoSpeculativeLoadHardening; }
};

class CLANG_ABI NoSplitStackAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_split_stack = 0,
    CXX11_gnu_no_split_stack = 1,
    C23_gnu_no_split_stack = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoSplitStackAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoSplitStackAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoSplitStackAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_split_stack);
  static NoSplitStackAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_split_stack);

  // Constructors
  NoSplitStackAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoSplitStackAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoSplitStack; }
};

class CLANG_ABI NoStackProtectorAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_stack_protector = 0,
    CXX11_clang_no_stack_protector = 1,
    C23_clang_no_stack_protector = 2,
    CXX11_gnu_no_stack_protector = 3,
    C23_gnu_no_stack_protector = 4,
    Declspec_safebuffers = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoStackProtectorAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoStackProtectorAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoStackProtectorAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_stack_protector);
  static NoStackProtectorAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_stack_protector);

  // Constructors
  NoStackProtectorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoStackProtectorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoStackProtector; }
};

class CLANG_ABI NoThreadSafetyAnalysisAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_no_thread_safety_analysis = 0,
    CXX11_clang_no_thread_safety_analysis = 1,
    C23_clang_no_thread_safety_analysis = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoThreadSafetyAnalysisAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoThreadSafetyAnalysisAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoThreadSafetyAnalysisAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_thread_safety_analysis);
  static NoThreadSafetyAnalysisAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_no_thread_safety_analysis);

  // Constructors
  NoThreadSafetyAnalysisAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoThreadSafetyAnalysisAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoThreadSafetyAnalysis; }
};

class CLANG_ABI NoThrowAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nothrow = 0,
    CXX11_gnu_nothrow = 1,
    C23_gnu_nothrow = 2,
    Declspec_nothrow = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoThrowAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoThrowAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoThrowAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nothrow);
  static NoThrowAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nothrow);

  // Constructors
  NoThrowAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoThrowAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoThrow; }
};

class CLANG_ABI NoTrivialAutoVarInitAttr : public InheritableAttr {
public:
  // Factory methods
  static NoTrivialAutoVarInitAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoTrivialAutoVarInitAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoTrivialAutoVarInitAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static NoTrivialAutoVarInitAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  NoTrivialAutoVarInitAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoTrivialAutoVarInitAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoTrivialAutoVarInit; }
};

class CLANG_ABI NoUniqueAddressAttr : public InheritableAttr {
public:
  enum Spelling {
    CXX11_no_unique_address = 0,
    CXX11_msvc_no_unique_address = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoUniqueAddressAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoUniqueAddressAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoUniqueAddressAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_no_unique_address);
  static NoUniqueAddressAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_no_unique_address);

  // Constructors
  NoUniqueAddressAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoUniqueAddressAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoUniqueAddress; }
};

class CLANG_ABI NoUwtableAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_nouwtable = 0,
    CXX11_clang_nouwtable = 1,
    C23_clang_nouwtable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NoUwtableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoUwtableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NoUwtableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nouwtable);
  static NoUwtableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_nouwtable);

  // Constructors
  NoUwtableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NoUwtableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NoUwtable; }
};

class CLANG_ABI NonAllocatingAttr : public TypeAttr {
Expr * cond;

public:
  enum Spelling {
    GNU_nonallocating = 0,
    CXX11_clang_nonallocating = 1,
    C23_clang_nonallocating = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NonAllocatingAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, const AttributeCommonInfo &CommonInfo);
  static NonAllocatingAttr *Create(ASTContext &Ctx, Expr * Cond, const AttributeCommonInfo &CommonInfo);
  static NonAllocatingAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, SourceRange Range = {}, Spelling S = GNU_nonallocating);
  static NonAllocatingAttr *Create(ASTContext &Ctx, Expr * Cond, SourceRange Range = {}, Spelling S = GNU_nonallocating);

  // Constructors
  NonAllocatingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
             );
  NonAllocatingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NonAllocatingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCond() const {
    return cond;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::NonAllocating; }
};

class CLANG_ABI NonBlockingAttr : public TypeAttr {
Expr * cond;

public:
  enum Spelling {
    GNU_nonblocking = 0,
    CXX11_clang_nonblocking = 1,
    C23_clang_nonblocking = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NonBlockingAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, const AttributeCommonInfo &CommonInfo);
  static NonBlockingAttr *Create(ASTContext &Ctx, Expr * Cond, const AttributeCommonInfo &CommonInfo);
  static NonBlockingAttr *CreateImplicit(ASTContext &Ctx, Expr * Cond, SourceRange Range = {}, Spelling S = GNU_nonblocking);
  static NonBlockingAttr *Create(ASTContext &Ctx, Expr * Cond, SourceRange Range = {}, Spelling S = GNU_nonblocking);

  // Constructors
  NonBlockingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Cond
             );
  NonBlockingAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NonBlockingAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getCond() const {
    return cond;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::NonBlocking; }
};

class CLANG_ABI NonNullAttr : public InheritableParamAttr {
  unsigned args_Size;
  ParamIdx *args_;

public:
  enum Spelling {
    GNU_nonnull = 0,
    CXX11_gnu_nonnull = 1,
    C23_gnu_nonnull = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NonNullAttr *CreateImplicit(ASTContext &Ctx, ParamIdx *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static NonNullAttr *Create(ASTContext &Ctx, ParamIdx *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static NonNullAttr *CreateImplicit(ASTContext &Ctx, ParamIdx *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_nonnull);
  static NonNullAttr *Create(ASTContext &Ctx, ParamIdx *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_nonnull);

  // Constructors
  NonNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamIdx *Args, unsigned ArgsSize
             );
  NonNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NonNullAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef ParamIdx* args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }



    bool isNonNull(unsigned IdxAST) const {
      if (!args_size())
        return true;
      return llvm::any_of(args(), [=](const ParamIdx &Idx) {
        return Idx.getASTIndex() == IdxAST;
      });
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::NonNull; }
};

class CLANG_ABI NotTailCalledAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_not_tail_called = 0,
    CXX11_clang_not_tail_called = 1,
    C23_clang_not_tail_called = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static NotTailCalledAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NotTailCalledAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static NotTailCalledAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_not_tail_called);
  static NotTailCalledAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_not_tail_called);

  // Constructors
  NotTailCalledAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  NotTailCalledAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::NotTailCalled; }
};

class CLANG_ABI OMPAllocateDeclAttr : public InheritableAttr {
public:
  enum AllocatorTypeTy {
    OMPNullMemAlloc,
    OMPDefaultMemAlloc,
    OMPLargeCapMemAlloc,
    OMPConstMemAlloc,
    OMPHighBWMemAlloc,
    OMPLowLatMemAlloc,
    OMPCGroupMemAlloc,
    OMPPTeamMemAlloc,
    OMPThreadMemAlloc,
    OMPUserDefinedMemAlloc
  };
private:
  OMPAllocateDeclAttr::AllocatorTypeTy allocatorType;

Expr * allocator;

Expr * alignment;

public:
  // Factory methods
  static OMPAllocateDeclAttr *CreateImplicit(ASTContext &Ctx, OMPAllocateDeclAttr::AllocatorTypeTy AllocatorType, Expr * Allocator, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static OMPAllocateDeclAttr *Create(ASTContext &Ctx, OMPAllocateDeclAttr::AllocatorTypeTy AllocatorType, Expr * Allocator, Expr * Alignment, const AttributeCommonInfo &CommonInfo);
  static OMPAllocateDeclAttr *CreateImplicit(ASTContext &Ctx, OMPAllocateDeclAttr::AllocatorTypeTy AllocatorType, Expr * Allocator, Expr * Alignment, SourceRange Range = {});
  static OMPAllocateDeclAttr *Create(ASTContext &Ctx, OMPAllocateDeclAttr::AllocatorTypeTy AllocatorType, Expr * Allocator, Expr * Alignment, SourceRange Range = {});

  // Constructors
  OMPAllocateDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , OMPAllocateDeclAttr::AllocatorTypeTy AllocatorType
              , Expr * Allocator
              , Expr * Alignment
             );

  OMPAllocateDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  OMPAllocateDeclAttr::AllocatorTypeTy getAllocatorType() const {
    return allocatorType;
  }

  static bool ConvertStrToAllocatorTypeTy(StringRef Val, OMPAllocateDeclAttr::AllocatorTypeTy &Out);
  static const char *ConvertAllocatorTypeTyToStr(OMPAllocateDeclAttr::AllocatorTypeTy Val);
  Expr * getAllocator() const {
    return allocator;
  }

  Expr * getAlignment() const {
    return alignment;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::OMPAllocateDecl; }
};

class CLANG_ABI OMPAssumeAttr : public InheritableAttr {
unsigned assumptionLength;
char *assumption;

public:
  // Factory methods
  static OMPAssumeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Assumption, const AttributeCommonInfo &CommonInfo);
  static OMPAssumeAttr *Create(ASTContext &Ctx, llvm::StringRef Assumption, const AttributeCommonInfo &CommonInfo);
  static OMPAssumeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Assumption, SourceRange Range = {});
  static OMPAssumeAttr *Create(ASTContext &Ctx, llvm::StringRef Assumption, SourceRange Range = {});

  // Constructors
  OMPAssumeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Assumption
             );

  OMPAssumeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAssumption() const {
    return llvm::StringRef(assumption, assumptionLength);
  }
  unsigned getAssumptionLength() const {
    return assumptionLength;
  }
  void setAssumption(ASTContext &C, llvm::StringRef S) {
    assumptionLength = S.size();
    this->assumption = new (C, 1) char [assumptionLength];
    if (!S.empty())
      std::memcpy(this->assumption, S.data(), assumptionLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::OMPAssume; }
};

class CLANG_ABI OMPCaptureKindAttr : public Attr {
unsigned captureKindVal;

public:
  // Factory methods
  static OMPCaptureKindAttr *CreateImplicit(ASTContext &Ctx, unsigned CaptureKindVal, const AttributeCommonInfo &CommonInfo);
  static OMPCaptureKindAttr *Create(ASTContext &Ctx, unsigned CaptureKindVal, const AttributeCommonInfo &CommonInfo);
  static OMPCaptureKindAttr *CreateImplicit(ASTContext &Ctx, unsigned CaptureKindVal, SourceRange Range = {});
  static OMPCaptureKindAttr *Create(ASTContext &Ctx, unsigned CaptureKindVal, SourceRange Range = {});

  // Constructors
  OMPCaptureKindAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned CaptureKindVal
             );

  OMPCaptureKindAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getCaptureKindVal() const {
    return captureKindVal;
  }


    llvm::omp::Clause getCaptureKind() const {
      return static_cast<llvm::omp::Clause>(getCaptureKindVal());
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::OMPCaptureKind; }
};

class CLANG_ABI OMPCaptureNoInitAttr : public InheritableAttr {
public:
  // Factory methods
  static OMPCaptureNoInitAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OMPCaptureNoInitAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OMPCaptureNoInitAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static OMPCaptureNoInitAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  OMPCaptureNoInitAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OMPCaptureNoInitAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OMPCaptureNoInit; }
};

class CLANG_ABI OMPDeclareSimdDeclAttr : public Attr {
public:
  enum BranchStateTy {
    BS_Undefined,
    BS_Inbranch,
    BS_Notinbranch
  };
private:
  OMPDeclareSimdDeclAttr::BranchStateTy branchState;

Expr * simdlen;

  unsigned uniforms_Size;
  Expr * *uniforms_;

  unsigned aligneds_Size;
  Expr * *aligneds_;

  unsigned alignments_Size;
  Expr * *alignments_;

  unsigned linears_Size;
  Expr * *linears_;

  unsigned modifiers_Size;
  unsigned *modifiers_;

  unsigned steps_Size;
  Expr * *steps_;

public:
  // Factory methods
  static OMPDeclareSimdDeclAttr *CreateImplicit(ASTContext &Ctx, OMPDeclareSimdDeclAttr::BranchStateTy BranchState, Expr * Simdlen, Expr * *Uniforms, unsigned UniformsSize, Expr * *Aligneds, unsigned AlignedsSize, Expr * *Alignments, unsigned AlignmentsSize, Expr * *Linears, unsigned LinearsSize, unsigned *Modifiers, unsigned ModifiersSize, Expr * *Steps, unsigned StepsSize, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareSimdDeclAttr *Create(ASTContext &Ctx, OMPDeclareSimdDeclAttr::BranchStateTy BranchState, Expr * Simdlen, Expr * *Uniforms, unsigned UniformsSize, Expr * *Aligneds, unsigned AlignedsSize, Expr * *Alignments, unsigned AlignmentsSize, Expr * *Linears, unsigned LinearsSize, unsigned *Modifiers, unsigned ModifiersSize, Expr * *Steps, unsigned StepsSize, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareSimdDeclAttr *CreateImplicit(ASTContext &Ctx, OMPDeclareSimdDeclAttr::BranchStateTy BranchState, Expr * Simdlen, Expr * *Uniforms, unsigned UniformsSize, Expr * *Aligneds, unsigned AlignedsSize, Expr * *Alignments, unsigned AlignmentsSize, Expr * *Linears, unsigned LinearsSize, unsigned *Modifiers, unsigned ModifiersSize, Expr * *Steps, unsigned StepsSize, SourceRange Range = {});
  static OMPDeclareSimdDeclAttr *Create(ASTContext &Ctx, OMPDeclareSimdDeclAttr::BranchStateTy BranchState, Expr * Simdlen, Expr * *Uniforms, unsigned UniformsSize, Expr * *Aligneds, unsigned AlignedsSize, Expr * *Alignments, unsigned AlignmentsSize, Expr * *Linears, unsigned LinearsSize, unsigned *Modifiers, unsigned ModifiersSize, Expr * *Steps, unsigned StepsSize, SourceRange Range = {});

  // Constructors
  OMPDeclareSimdDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , OMPDeclareSimdDeclAttr::BranchStateTy BranchState
              , Expr * Simdlen
              , Expr * *Uniforms, unsigned UniformsSize
              , Expr * *Aligneds, unsigned AlignedsSize
              , Expr * *Alignments, unsigned AlignmentsSize
              , Expr * *Linears, unsigned LinearsSize
              , unsigned *Modifiers, unsigned ModifiersSize
              , Expr * *Steps, unsigned StepsSize
             );
  OMPDeclareSimdDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , OMPDeclareSimdDeclAttr::BranchStateTy BranchState
              , Expr * Simdlen
             );

  OMPDeclareSimdDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  OMPDeclareSimdDeclAttr::BranchStateTy getBranchState() const {
    return branchState;
  }

  static bool ConvertStrToBranchStateTy(StringRef Val, OMPDeclareSimdDeclAttr::BranchStateTy &Out);
  static const char *ConvertBranchStateTyToStr(OMPDeclareSimdDeclAttr::BranchStateTy Val);
  Expr * getSimdlen() const {
    return simdlen;
  }

  typedef Expr ** uniforms_iterator;
  uniforms_iterator uniforms_begin() const { return uniforms_; }
  uniforms_iterator uniforms_end() const { return uniforms_ + uniforms_Size; }
  unsigned uniforms_size() const { return uniforms_Size; }
  llvm::iterator_range<uniforms_iterator> uniforms() const { return llvm::make_range(uniforms_begin(), uniforms_end()); }


  typedef Expr ** aligneds_iterator;
  aligneds_iterator aligneds_begin() const { return aligneds_; }
  aligneds_iterator aligneds_end() const { return aligneds_ + aligneds_Size; }
  unsigned aligneds_size() const { return aligneds_Size; }
  llvm::iterator_range<aligneds_iterator> aligneds() const { return llvm::make_range(aligneds_begin(), aligneds_end()); }


  typedef Expr ** alignments_iterator;
  alignments_iterator alignments_begin() const { return alignments_; }
  alignments_iterator alignments_end() const { return alignments_ + alignments_Size; }
  unsigned alignments_size() const { return alignments_Size; }
  llvm::iterator_range<alignments_iterator> alignments() const { return llvm::make_range(alignments_begin(), alignments_end()); }


  typedef Expr ** linears_iterator;
  linears_iterator linears_begin() const { return linears_; }
  linears_iterator linears_end() const { return linears_ + linears_Size; }
  unsigned linears_size() const { return linears_Size; }
  llvm::iterator_range<linears_iterator> linears() const { return llvm::make_range(linears_begin(), linears_end()); }


  typedef unsigned* modifiers_iterator;
  modifiers_iterator modifiers_begin() const { return modifiers_; }
  modifiers_iterator modifiers_end() const { return modifiers_ + modifiers_Size; }
  unsigned modifiers_size() const { return modifiers_Size; }
  llvm::iterator_range<modifiers_iterator> modifiers() const { return llvm::make_range(modifiers_begin(), modifiers_end()); }


  typedef Expr ** steps_iterator;
  steps_iterator steps_begin() const { return steps_; }
  steps_iterator steps_end() const { return steps_ + steps_Size; }
  unsigned steps_size() const { return steps_Size; }
  llvm::iterator_range<steps_iterator> steps() const { return llvm::make_range(steps_begin(), steps_end()); }



    void printPrettyPragma(raw_ostream & OS, const PrintingPolicy &Policy)
        const;
  

  static bool classof(const Attr *A) { return A->getKind() == attr::OMPDeclareSimdDecl; }
};

class CLANG_ABI OMPDeclareTargetDeclAttr : public InheritableAttr {
public:
  enum MapTypeTy {
    MT_To,
    MT_Enter,
    MT_Link
  };
private:
  OMPDeclareTargetDeclAttr::MapTypeTy mapType;

public:
  enum DevTypeTy {
    DT_Host,
    DT_NoHost,
    DT_Any
  };
private:
  OMPDeclareTargetDeclAttr::DevTypeTy devType;

Expr * indirectExpr;

bool indirect;

unsigned level;

public:
  // Factory methods
  static OMPDeclareTargetDeclAttr *CreateImplicit(ASTContext &Ctx, OMPDeclareTargetDeclAttr::MapTypeTy MapType, OMPDeclareTargetDeclAttr::DevTypeTy DevType, Expr * IndirectExpr, bool Indirect, unsigned Level, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareTargetDeclAttr *Create(ASTContext &Ctx, OMPDeclareTargetDeclAttr::MapTypeTy MapType, OMPDeclareTargetDeclAttr::DevTypeTy DevType, Expr * IndirectExpr, bool Indirect, unsigned Level, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareTargetDeclAttr *CreateImplicit(ASTContext &Ctx, OMPDeclareTargetDeclAttr::MapTypeTy MapType, OMPDeclareTargetDeclAttr::DevTypeTy DevType, Expr * IndirectExpr, bool Indirect, unsigned Level, SourceRange Range = {});
  static OMPDeclareTargetDeclAttr *Create(ASTContext &Ctx, OMPDeclareTargetDeclAttr::MapTypeTy MapType, OMPDeclareTargetDeclAttr::DevTypeTy DevType, Expr * IndirectExpr, bool Indirect, unsigned Level, SourceRange Range = {});

  // Constructors
  OMPDeclareTargetDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , OMPDeclareTargetDeclAttr::MapTypeTy MapType
              , OMPDeclareTargetDeclAttr::DevTypeTy DevType
              , Expr * IndirectExpr
              , bool Indirect
              , unsigned Level
             );

  OMPDeclareTargetDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  OMPDeclareTargetDeclAttr::MapTypeTy getMapType() const {
    return mapType;
  }

  static bool ConvertStrToMapTypeTy(StringRef Val, OMPDeclareTargetDeclAttr::MapTypeTy &Out);
  static const char *ConvertMapTypeTyToStr(OMPDeclareTargetDeclAttr::MapTypeTy Val);
  OMPDeclareTargetDeclAttr::DevTypeTy getDevType() const {
    return devType;
  }

  static bool ConvertStrToDevTypeTy(StringRef Val, OMPDeclareTargetDeclAttr::DevTypeTy &Out);
  static const char *ConvertDevTypeTyToStr(OMPDeclareTargetDeclAttr::DevTypeTy Val);
  Expr * getIndirectExpr() const {
    return indirectExpr;
  }

  bool getIndirect() const {
    return indirect;
  }

  unsigned getLevel() const {
    return level;
  }


    void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const;
    static std::optional<MapTypeTy>
    isDeclareTargetDeclaration(const ValueDecl *VD);
    static std::optional<OMPDeclareTargetDeclAttr*> getActiveAttr(const ValueDecl *VD);
    static std::optional<DevTypeTy> getDeviceType(const ValueDecl *VD);
    static std::optional<SourceLocation> getLocation(const ValueDecl *VD);
  

  static bool classof(const Attr *A) { return A->getKind() == attr::OMPDeclareTargetDecl; }
};

class CLANG_ABI OMPDeclareVariantAttr : public InheritableAttr {
Expr * variantFuncRef;

OMPTraitInfo * traitInfos;

  unsigned adjustArgsNothing_Size;
  Expr * *adjustArgsNothing_;

  unsigned adjustArgsNeedDevicePtr_Size;
  Expr * *adjustArgsNeedDevicePtr_;

  unsigned appendArgs_Size;
  OMPInteropInfo *appendArgs_;

public:
  // Factory methods
  static OMPDeclareVariantAttr *CreateImplicit(ASTContext &Ctx, Expr * VariantFuncRef, OMPTraitInfo * TraitInfos, Expr * *AdjustArgsNothing, unsigned AdjustArgsNothingSize, Expr * *AdjustArgsNeedDevicePtr, unsigned AdjustArgsNeedDevicePtrSize, OMPInteropInfo *AppendArgs, unsigned AppendArgsSize, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareVariantAttr *Create(ASTContext &Ctx, Expr * VariantFuncRef, OMPTraitInfo * TraitInfos, Expr * *AdjustArgsNothing, unsigned AdjustArgsNothingSize, Expr * *AdjustArgsNeedDevicePtr, unsigned AdjustArgsNeedDevicePtrSize, OMPInteropInfo *AppendArgs, unsigned AppendArgsSize, const AttributeCommonInfo &CommonInfo);
  static OMPDeclareVariantAttr *CreateImplicit(ASTContext &Ctx, Expr * VariantFuncRef, OMPTraitInfo * TraitInfos, Expr * *AdjustArgsNothing, unsigned AdjustArgsNothingSize, Expr * *AdjustArgsNeedDevicePtr, unsigned AdjustArgsNeedDevicePtrSize, OMPInteropInfo *AppendArgs, unsigned AppendArgsSize, SourceRange Range = {});
  static OMPDeclareVariantAttr *Create(ASTContext &Ctx, Expr * VariantFuncRef, OMPTraitInfo * TraitInfos, Expr * *AdjustArgsNothing, unsigned AdjustArgsNothingSize, Expr * *AdjustArgsNeedDevicePtr, unsigned AdjustArgsNeedDevicePtrSize, OMPInteropInfo *AppendArgs, unsigned AppendArgsSize, SourceRange Range = {});

  // Constructors
  OMPDeclareVariantAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * VariantFuncRef
              , OMPTraitInfo * TraitInfos
              , Expr * *AdjustArgsNothing, unsigned AdjustArgsNothingSize
              , Expr * *AdjustArgsNeedDevicePtr, unsigned AdjustArgsNeedDevicePtrSize
              , OMPInteropInfo *AppendArgs, unsigned AppendArgsSize
             );
  OMPDeclareVariantAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * VariantFuncRef
              , OMPTraitInfo * TraitInfos
              , OMPInteropInfo *AppendArgs, unsigned AppendArgsSize
             );

  OMPDeclareVariantAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getVariantFuncRef() const {
    return variantFuncRef;
  }

  OMPTraitInfo * getTraitInfos() const {
    return traitInfos;
  }

  typedef Expr ** adjustArgsNothing_iterator;
  adjustArgsNothing_iterator adjustArgsNothing_begin() const { return adjustArgsNothing_; }
  adjustArgsNothing_iterator adjustArgsNothing_end() const { return adjustArgsNothing_ + adjustArgsNothing_Size; }
  unsigned adjustArgsNothing_size() const { return adjustArgsNothing_Size; }
  llvm::iterator_range<adjustArgsNothing_iterator> adjustArgsNothing() const { return llvm::make_range(adjustArgsNothing_begin(), adjustArgsNothing_end()); }


  typedef Expr ** adjustArgsNeedDevicePtr_iterator;
  adjustArgsNeedDevicePtr_iterator adjustArgsNeedDevicePtr_begin() const { return adjustArgsNeedDevicePtr_; }
  adjustArgsNeedDevicePtr_iterator adjustArgsNeedDevicePtr_end() const { return adjustArgsNeedDevicePtr_ + adjustArgsNeedDevicePtr_Size; }
  unsigned adjustArgsNeedDevicePtr_size() const { return adjustArgsNeedDevicePtr_Size; }
  llvm::iterator_range<adjustArgsNeedDevicePtr_iterator> adjustArgsNeedDevicePtr() const { return llvm::make_range(adjustArgsNeedDevicePtr_begin(), adjustArgsNeedDevicePtr_end()); }


  typedef OMPInteropInfo* appendArgs_iterator;
  appendArgs_iterator appendArgs_begin() const { return appendArgs_; }
  appendArgs_iterator appendArgs_end() const { return appendArgs_ + appendArgs_Size; }
  unsigned appendArgs_size() const { return appendArgs_Size; }
  llvm::iterator_range<appendArgs_iterator> appendArgs() const { return llvm::make_range(appendArgs_begin(), appendArgs_end()); }



    OMPTraitInfo &getTraitInfo() { return *traitInfos; }
    void printPrettyPragma(raw_ostream & OS, const PrintingPolicy &Policy)
        const;
    static StringRef getInteropTypeString(const OMPInteropInfo *I) {
      if (I->IsTarget && I->IsTargetSync)
        return "target,targetsync";
      if (I->IsTarget)
        return "target";
      return "targetsync";
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::OMPDeclareVariant; }
};

class CLANG_ABI OMPReferencedVarAttr : public Attr {
Expr * ref;

public:
  // Factory methods
  static OMPReferencedVarAttr *CreateImplicit(ASTContext &Ctx, Expr * Ref, const AttributeCommonInfo &CommonInfo);
  static OMPReferencedVarAttr *Create(ASTContext &Ctx, Expr * Ref, const AttributeCommonInfo &CommonInfo);
  static OMPReferencedVarAttr *CreateImplicit(ASTContext &Ctx, Expr * Ref, SourceRange Range = {});
  static OMPReferencedVarAttr *Create(ASTContext &Ctx, Expr * Ref, SourceRange Range = {});

  // Constructors
  OMPReferencedVarAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Ref
             );

  OMPReferencedVarAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getRef() const {
    return ref;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::OMPReferencedVar; }
};

class CLANG_ABI OMPThreadPrivateDeclAttr : public InheritableAttr {
public:
  // Factory methods
  static OMPThreadPrivateDeclAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OMPThreadPrivateDeclAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OMPThreadPrivateDeclAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static OMPThreadPrivateDeclAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  OMPThreadPrivateDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OMPThreadPrivateDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OMPThreadPrivateDecl; }
};

class CLANG_ABI OSConsumedAttr : public InheritableParamAttr {
public:
  enum Spelling {
    GNU_os_consumed = 0,
    CXX11_clang_os_consumed = 1,
    C23_clang_os_consumed = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSConsumedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSConsumedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSConsumedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_consumed);
  static OSConsumedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_consumed);

  // Constructors
  OSConsumedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSConsumedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSConsumed; }
};

class CLANG_ABI OSConsumesThisAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_os_consumes_this = 0,
    CXX11_clang_os_consumes_this = 1,
    C23_clang_os_consumes_this = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSConsumesThisAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSConsumesThisAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSConsumesThisAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_consumes_this);
  static OSConsumesThisAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_consumes_this);

  // Constructors
  OSConsumesThisAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSConsumesThisAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSConsumesThis; }
};

class CLANG_ABI OSReturnsNotRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_os_returns_not_retained = 0,
    CXX11_clang_os_returns_not_retained = 1,
    C23_clang_os_returns_not_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsNotRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsNotRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_not_retained);
  static OSReturnsNotRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_not_retained);

  // Constructors
  OSReturnsNotRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSReturnsNotRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSReturnsNotRetained; }
};

class CLANG_ABI OSReturnsRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_os_returns_retained = 0,
    CXX11_clang_os_returns_retained = 1,
    C23_clang_os_returns_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained);
  static OSReturnsRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained);

  // Constructors
  OSReturnsRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSReturnsRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSReturnsRetained; }
};

class CLANG_ABI OSReturnsRetainedOnNonZeroAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_os_returns_retained_on_non_zero = 0,
    CXX11_clang_os_returns_retained_on_non_zero = 1,
    C23_clang_os_returns_retained_on_non_zero = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSReturnsRetainedOnNonZeroAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedOnNonZeroAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedOnNonZeroAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained_on_non_zero);
  static OSReturnsRetainedOnNonZeroAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained_on_non_zero);

  // Constructors
  OSReturnsRetainedOnNonZeroAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSReturnsRetainedOnNonZeroAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSReturnsRetainedOnNonZero; }
};

class CLANG_ABI OSReturnsRetainedOnZeroAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_os_returns_retained_on_zero = 0,
    CXX11_clang_os_returns_retained_on_zero = 1,
    C23_clang_os_returns_retained_on_zero = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OSReturnsRetainedOnZeroAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedOnZeroAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OSReturnsRetainedOnZeroAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained_on_zero);
  static OSReturnsRetainedOnZeroAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_os_returns_retained_on_zero);

  // Constructors
  OSReturnsRetainedOnZeroAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OSReturnsRetainedOnZeroAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OSReturnsRetainedOnZero; }
};

class CLANG_ABI ObjCBoxableAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_boxable = 0,
    CXX11_clang_objc_boxable = 1,
    C23_clang_objc_boxable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCBoxableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCBoxableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCBoxableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_boxable);
  static ObjCBoxableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_boxable);

  // Constructors
  ObjCBoxableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCBoxableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCBoxable; }
};

class CLANG_ABI ObjCBridgeAttr : public InheritableAttr {
IdentifierInfo * bridgedType;

public:
  enum Spelling {
    GNU_objc_bridge = 0,
    CXX11_clang_objc_bridge = 1,
    C23_clang_objc_bridge = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCBridgeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BridgedType, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeAttr *Create(ASTContext &Ctx, IdentifierInfo * BridgedType, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BridgedType, SourceRange Range = {}, Spelling S = GNU_objc_bridge);
  static ObjCBridgeAttr *Create(ASTContext &Ctx, IdentifierInfo * BridgedType, SourceRange Range = {}, Spelling S = GNU_objc_bridge);

  // Constructors
  ObjCBridgeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * BridgedType
             );

  ObjCBridgeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getBridgedType() const {
    return bridgedType;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCBridge; }
};

class CLANG_ABI ObjCBridgeMutableAttr : public InheritableAttr {
IdentifierInfo * bridgedType;

public:
  enum Spelling {
    GNU_objc_bridge_mutable = 0,
    CXX11_clang_objc_bridge_mutable = 1,
    C23_clang_objc_bridge_mutable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCBridgeMutableAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BridgedType, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeMutableAttr *Create(ASTContext &Ctx, IdentifierInfo * BridgedType, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeMutableAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * BridgedType, SourceRange Range = {}, Spelling S = GNU_objc_bridge_mutable);
  static ObjCBridgeMutableAttr *Create(ASTContext &Ctx, IdentifierInfo * BridgedType, SourceRange Range = {}, Spelling S = GNU_objc_bridge_mutable);

  // Constructors
  ObjCBridgeMutableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * BridgedType
             );

  ObjCBridgeMutableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getBridgedType() const {
    return bridgedType;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCBridgeMutable; }
};

class CLANG_ABI ObjCBridgeRelatedAttr : public InheritableAttr {
IdentifierInfo * relatedClass;

IdentifierInfo * classMethod;

IdentifierInfo * instanceMethod;

public:
  enum Spelling {
    GNU_objc_bridge_related = 0,
    CXX11_clang_objc_bridge_related = 1,
    C23_clang_objc_bridge_related = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCBridgeRelatedAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * RelatedClass, IdentifierInfo * ClassMethod, IdentifierInfo * InstanceMethod, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeRelatedAttr *Create(ASTContext &Ctx, IdentifierInfo * RelatedClass, IdentifierInfo * ClassMethod, IdentifierInfo * InstanceMethod, const AttributeCommonInfo &CommonInfo);
  static ObjCBridgeRelatedAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * RelatedClass, IdentifierInfo * ClassMethod, IdentifierInfo * InstanceMethod, SourceRange Range = {}, Spelling S = GNU_objc_bridge_related);
  static ObjCBridgeRelatedAttr *Create(ASTContext &Ctx, IdentifierInfo * RelatedClass, IdentifierInfo * ClassMethod, IdentifierInfo * InstanceMethod, SourceRange Range = {}, Spelling S = GNU_objc_bridge_related);

  // Constructors
  ObjCBridgeRelatedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * RelatedClass
              , IdentifierInfo * ClassMethod
              , IdentifierInfo * InstanceMethod
             );

  ObjCBridgeRelatedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getRelatedClass() const {
    return relatedClass;
  }

  IdentifierInfo * getClassMethod() const {
    return classMethod;
  }

  IdentifierInfo * getInstanceMethod() const {
    return instanceMethod;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCBridgeRelated; }
};

class CLANG_ABI ObjCClassStubAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_class_stub = 0,
    CXX11_clang_objc_class_stub = 1,
    C23_clang_objc_class_stub = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCClassStubAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCClassStubAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCClassStubAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_class_stub);
  static ObjCClassStubAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_class_stub);

  // Constructors
  ObjCClassStubAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCClassStubAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCClassStub; }
};

class CLANG_ABI ObjCDesignatedInitializerAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_designated_initializer = 0,
    CXX11_clang_objc_designated_initializer = 1,
    C23_clang_objc_designated_initializer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCDesignatedInitializerAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDesignatedInitializerAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDesignatedInitializerAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_designated_initializer);
  static ObjCDesignatedInitializerAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_designated_initializer);

  // Constructors
  ObjCDesignatedInitializerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCDesignatedInitializerAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCDesignatedInitializer; }
};

class CLANG_ABI ObjCDirectAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_direct = 0,
    CXX11_clang_objc_direct = 1,
    C23_clang_objc_direct = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCDirectAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDirectAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDirectAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_direct);
  static ObjCDirectAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_direct);

  // Constructors
  ObjCDirectAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCDirectAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCDirect; }
};

class CLANG_ABI ObjCDirectMembersAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_direct_members = 0,
    CXX11_clang_objc_direct_members = 1,
    C23_clang_objc_direct_members = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCDirectMembersAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDirectMembersAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCDirectMembersAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_direct_members);
  static ObjCDirectMembersAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_direct_members);

  // Constructors
  ObjCDirectMembersAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCDirectMembersAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCDirectMembers; }
};

class CLANG_ABI ObjCExceptionAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_exception = 0,
    CXX11_clang_objc_exception = 1,
    C23_clang_objc_exception = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCExceptionAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExceptionAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExceptionAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_exception);
  static ObjCExceptionAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_exception);

  // Constructors
  ObjCExceptionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCExceptionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCException; }
};

class CLANG_ABI ObjCExplicitProtocolImplAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_protocol_requires_explicit_implementation = 0,
    CXX11_clang_objc_protocol_requires_explicit_implementation = 1,
    C23_clang_objc_protocol_requires_explicit_implementation = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCExplicitProtocolImplAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExplicitProtocolImplAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExplicitProtocolImplAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_protocol_requires_explicit_implementation);
  static ObjCExplicitProtocolImplAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_protocol_requires_explicit_implementation);

  // Constructors
  ObjCExplicitProtocolImplAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCExplicitProtocolImplAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCExplicitProtocolImpl; }
};

class CLANG_ABI ObjCExternallyRetainedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_externally_retained = 0,
    CXX11_clang_objc_externally_retained = 1,
    C23_clang_objc_externally_retained = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCExternallyRetainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExternallyRetainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCExternallyRetainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_externally_retained);
  static ObjCExternallyRetainedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_externally_retained);

  // Constructors
  ObjCExternallyRetainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCExternallyRetainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCExternallyRetained; }
};

class CLANG_ABI ObjCGCAttr : public TypeAttr {
IdentifierInfo * kind;

public:
  enum Spelling {
    GNU_objc_gc = 0,
    CXX11_clang_objc_gc = 1,
    C23_clang_objc_gc = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCGCAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Kind, const AttributeCommonInfo &CommonInfo);
  static ObjCGCAttr *Create(ASTContext &Ctx, IdentifierInfo * Kind, const AttributeCommonInfo &CommonInfo);
  static ObjCGCAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Kind, SourceRange Range = {}, Spelling S = GNU_objc_gc);
  static ObjCGCAttr *Create(ASTContext &Ctx, IdentifierInfo * Kind, SourceRange Range = {}, Spelling S = GNU_objc_gc);

  // Constructors
  ObjCGCAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Kind
             );

  ObjCGCAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getKind() const {
    return kind;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCGC; }
};

class CLANG_ABI ObjCIndependentClassAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_independent_class = 0,
    CXX11_clang_objc_independent_class = 1,
    C23_clang_objc_independent_class = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCIndependentClassAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCIndependentClassAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCIndependentClassAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_independent_class);
  static ObjCIndependentClassAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_independent_class);

  // Constructors
  ObjCIndependentClassAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCIndependentClassAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCIndependentClass; }
};

class CLANG_ABI ObjCInertUnsafeUnretainedAttr : public TypeAttr {
public:
  // Factory methods
  static ObjCInertUnsafeUnretainedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCInertUnsafeUnretainedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCInertUnsafeUnretainedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ObjCInertUnsafeUnretainedAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ObjCInertUnsafeUnretainedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCInertUnsafeUnretainedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCInertUnsafeUnretained; }
};

class CLANG_ABI ObjCKindOfAttr : public TypeAttr {
public:
  // Factory methods
  static ObjCKindOfAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCKindOfAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCKindOfAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ObjCKindOfAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ObjCKindOfAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCKindOfAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCKindOf; }
};

class CLANG_ABI ObjCMethodFamilyAttr : public InheritableAttr {
public:
  enum FamilyKind {
    OMF_None,
    OMF_alloc,
    OMF_copy,
    OMF_init,
    OMF_mutableCopy,
    OMF_new
  };
private:
  ObjCMethodFamilyAttr::FamilyKind family;

public:
  enum Spelling {
    GNU_objc_method_family = 0,
    CXX11_clang_objc_method_family = 1,
    C23_clang_objc_method_family = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCMethodFamilyAttr *CreateImplicit(ASTContext &Ctx, ObjCMethodFamilyAttr::FamilyKind Family, const AttributeCommonInfo &CommonInfo);
  static ObjCMethodFamilyAttr *Create(ASTContext &Ctx, ObjCMethodFamilyAttr::FamilyKind Family, const AttributeCommonInfo &CommonInfo);
  static ObjCMethodFamilyAttr *CreateImplicit(ASTContext &Ctx, ObjCMethodFamilyAttr::FamilyKind Family, SourceRange Range = {}, Spelling S = GNU_objc_method_family);
  static ObjCMethodFamilyAttr *Create(ASTContext &Ctx, ObjCMethodFamilyAttr::FamilyKind Family, SourceRange Range = {}, Spelling S = GNU_objc_method_family);

  // Constructors
  ObjCMethodFamilyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ObjCMethodFamilyAttr::FamilyKind Family
             );

  ObjCMethodFamilyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ObjCMethodFamilyAttr::FamilyKind getFamily() const {
    return family;
  }

  static bool ConvertStrToFamilyKind(StringRef Val, ObjCMethodFamilyAttr::FamilyKind &Out);
  static const char *ConvertFamilyKindToStr(ObjCMethodFamilyAttr::FamilyKind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCMethodFamily; }
};

class CLANG_ABI ObjCNSObjectAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_NSObject = 0,
    CXX11_clang_NSObject = 1,
    C23_clang_NSObject = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCNSObjectAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNSObjectAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNSObjectAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_NSObject);
  static ObjCNSObjectAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_NSObject);

  // Constructors
  ObjCNSObjectAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCNSObjectAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCNSObject; }
};

class CLANG_ABI ObjCNonLazyClassAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_nonlazy_class = 0,
    CXX11_clang_objc_nonlazy_class = 1,
    C23_clang_objc_nonlazy_class = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCNonLazyClassAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNonLazyClassAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNonLazyClassAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_nonlazy_class);
  static ObjCNonLazyClassAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_nonlazy_class);

  // Constructors
  ObjCNonLazyClassAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCNonLazyClassAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCNonLazyClass; }
};

class CLANG_ABI ObjCNonRuntimeProtocolAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_non_runtime_protocol = 0,
    CXX11_clang_objc_non_runtime_protocol = 1,
    C23_clang_objc_non_runtime_protocol = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCNonRuntimeProtocolAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNonRuntimeProtocolAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCNonRuntimeProtocolAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_non_runtime_protocol);
  static ObjCNonRuntimeProtocolAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_non_runtime_protocol);

  // Constructors
  ObjCNonRuntimeProtocolAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCNonRuntimeProtocolAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCNonRuntimeProtocol; }
};

class CLANG_ABI ObjCOwnershipAttr : public InheritableAttr {
IdentifierInfo * kind;

public:
  enum Spelling {
    GNU_objc_ownership = 0,
    CXX11_clang_objc_ownership = 1,
    C23_clang_objc_ownership = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCOwnershipAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Kind, const AttributeCommonInfo &CommonInfo);
  static ObjCOwnershipAttr *Create(ASTContext &Ctx, IdentifierInfo * Kind, const AttributeCommonInfo &CommonInfo);
  static ObjCOwnershipAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Kind, SourceRange Range = {}, Spelling S = GNU_objc_ownership);
  static ObjCOwnershipAttr *Create(ASTContext &Ctx, IdentifierInfo * Kind, SourceRange Range = {}, Spelling S = GNU_objc_ownership);

  // Constructors
  ObjCOwnershipAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Kind
             );

  ObjCOwnershipAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getKind() const {
    return kind;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCOwnership; }
};

class CLANG_ABI ObjCPreciseLifetimeAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_precise_lifetime = 0,
    CXX11_clang_objc_precise_lifetime = 1,
    C23_clang_objc_precise_lifetime = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCPreciseLifetimeAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCPreciseLifetimeAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCPreciseLifetimeAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_precise_lifetime);
  static ObjCPreciseLifetimeAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_precise_lifetime);

  // Constructors
  ObjCPreciseLifetimeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCPreciseLifetimeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCPreciseLifetime; }
};

class CLANG_ABI ObjCRequiresPropertyDefsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_requires_property_definitions = 0,
    CXX11_clang_objc_requires_property_definitions = 1,
    C23_clang_objc_requires_property_definitions = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCRequiresPropertyDefsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRequiresPropertyDefsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRequiresPropertyDefsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_requires_property_definitions);
  static ObjCRequiresPropertyDefsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_requires_property_definitions);

  // Constructors
  ObjCRequiresPropertyDefsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCRequiresPropertyDefsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCRequiresPropertyDefs; }
};

class CLANG_ABI ObjCRequiresSuperAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_requires_super = 0,
    CXX11_clang_objc_requires_super = 1,
    C23_clang_objc_requires_super = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCRequiresSuperAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRequiresSuperAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRequiresSuperAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_requires_super);
  static ObjCRequiresSuperAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_requires_super);

  // Constructors
  ObjCRequiresSuperAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCRequiresSuperAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCRequiresSuper; }
};

class CLANG_ABI ObjCReturnsInnerPointerAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_returns_inner_pointer = 0,
    CXX11_clang_objc_returns_inner_pointer = 1,
    C23_clang_objc_returns_inner_pointer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCReturnsInnerPointerAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCReturnsInnerPointerAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCReturnsInnerPointerAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_returns_inner_pointer);
  static ObjCReturnsInnerPointerAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_returns_inner_pointer);

  // Constructors
  ObjCReturnsInnerPointerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCReturnsInnerPointerAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCReturnsInnerPointer; }
};

class CLANG_ABI ObjCRootClassAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_root_class = 0,
    CXX11_clang_objc_root_class = 1,
    C23_clang_objc_root_class = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCRootClassAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRootClassAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRootClassAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_root_class);
  static ObjCRootClassAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_root_class);

  // Constructors
  ObjCRootClassAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCRootClassAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCRootClass; }
};

class CLANG_ABI ObjCRuntimeNameAttr : public Attr {
unsigned metadataNameLength;
char *metadataName;

public:
  enum Spelling {
    GNU_objc_runtime_name = 0,
    CXX11_clang_objc_runtime_name = 1,
    C23_clang_objc_runtime_name = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCRuntimeNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef MetadataName, const AttributeCommonInfo &CommonInfo);
  static ObjCRuntimeNameAttr *Create(ASTContext &Ctx, llvm::StringRef MetadataName, const AttributeCommonInfo &CommonInfo);
  static ObjCRuntimeNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef MetadataName, SourceRange Range = {}, Spelling S = GNU_objc_runtime_name);
  static ObjCRuntimeNameAttr *Create(ASTContext &Ctx, llvm::StringRef MetadataName, SourceRange Range = {}, Spelling S = GNU_objc_runtime_name);

  // Constructors
  ObjCRuntimeNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef MetadataName
             );

  ObjCRuntimeNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getMetadataName() const {
    return llvm::StringRef(metadataName, metadataNameLength);
  }
  unsigned getMetadataNameLength() const {
    return metadataNameLength;
  }
  void setMetadataName(ASTContext &C, llvm::StringRef S) {
    metadataNameLength = S.size();
    this->metadataName = new (C, 1) char [metadataNameLength];
    if (!S.empty())
      std::memcpy(this->metadataName, S.data(), metadataNameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCRuntimeName; }
};

class CLANG_ABI ObjCRuntimeVisibleAttr : public Attr {
public:
  enum Spelling {
    GNU_objc_runtime_visible = 0,
    CXX11_clang_objc_runtime_visible = 1,
    C23_clang_objc_runtime_visible = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCRuntimeVisibleAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRuntimeVisibleAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCRuntimeVisibleAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_runtime_visible);
  static ObjCRuntimeVisibleAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_runtime_visible);

  // Constructors
  ObjCRuntimeVisibleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCRuntimeVisibleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCRuntimeVisible; }
};

class CLANG_ABI ObjCSubclassingRestrictedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_objc_subclassing_restricted = 0,
    CXX11_clang_objc_subclassing_restricted = 1,
    C23_clang_objc_subclassing_restricted = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ObjCSubclassingRestrictedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCSubclassingRestrictedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ObjCSubclassingRestrictedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_subclassing_restricted);
  static ObjCSubclassingRestrictedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_objc_subclassing_restricted);

  // Constructors
  ObjCSubclassingRestrictedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ObjCSubclassingRestrictedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ObjCSubclassingRestricted; }
};

class CLANG_ABI OpenACCRoutineAnnotAttr : public InheritableAttr {
public:
  // Factory methods
  static OpenACCRoutineAnnotAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenACCRoutineAnnotAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenACCRoutineAnnotAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static OpenACCRoutineAnnotAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  OpenACCRoutineAnnotAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenACCRoutineAnnotAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenACCRoutineAnnot; }
};

class CLANG_ABI OpenACCRoutineDeclAttr : public InheritableAttr {
public:
  // Factory methods
  static OpenACCRoutineDeclAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenACCRoutineDeclAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenACCRoutineDeclAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static OpenACCRoutineDeclAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  OpenACCRoutineDeclAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenACCRoutineDeclAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;

    llvm::SmallVector<const OpenACCClause *> Clauses;
    void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const;
  

  static bool classof(const Attr *A) { return A->getKind() == attr::OpenACCRoutineDecl; }
};

class CLANG_ABI OpenCLAccessAttr : public Attr {
public:
  enum Spelling {
    Keyword_read_only = 0,
    Keyword_write_only = 2,
    Keyword_read_write = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLAccessAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLAccessAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLAccessAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_read_only);
  static OpenCLAccessAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_read_only);

  // Constructors
  OpenCLAccessAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLAccessAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isReadOnly() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1; }
  bool isReadWrite() const { return getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }
  bool isWriteOnly() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3; }


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLAccess; }
};

class CLANG_ABI OpenCLConstantAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    Keyword_constant = 0,
    GNU_opencl_constant = 2,
    CXX11_clang_opencl_constant = 3,
    C23_clang_opencl_constant = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLConstantAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLConstantAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLConstantAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_constant);
  static OpenCLConstantAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_constant);

  // Constructors
  OpenCLConstantAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLConstantAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLConstantAddressSpace; }
};

class CLANG_ABI OpenCLGenericAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    Keyword_generic = 0,
    GNU_opencl_generic = 2,
    CXX11_clang_opencl_generic = 3,
    C23_clang_opencl_generic = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLGenericAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGenericAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGenericAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_generic);
  static OpenCLGenericAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_generic);

  // Constructors
  OpenCLGenericAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLGenericAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLGenericAddressSpace; }
};

class CLANG_ABI OpenCLGlobalAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    Keyword_global = 0,
    GNU_opencl_global = 2,
    CXX11_clang_opencl_global = 3,
    C23_clang_opencl_global = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLGlobalAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_global);
  static OpenCLGlobalAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_global);

  // Constructors
  OpenCLGlobalAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLGlobalAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLGlobalAddressSpace; }
};

class CLANG_ABI OpenCLGlobalDeviceAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_opencl_global_device = 0,
    CXX11_clang_opencl_global_device = 1,
    C23_clang_opencl_global_device = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLGlobalDeviceAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalDeviceAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalDeviceAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_opencl_global_device);
  static OpenCLGlobalDeviceAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_opencl_global_device);

  // Constructors
  OpenCLGlobalDeviceAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLGlobalDeviceAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLGlobalDeviceAddressSpace; }
};

class CLANG_ABI OpenCLGlobalHostAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    GNU_opencl_global_host = 0,
    CXX11_clang_opencl_global_host = 1,
    C23_clang_opencl_global_host = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLGlobalHostAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalHostAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLGlobalHostAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_opencl_global_host);
  static OpenCLGlobalHostAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_opencl_global_host);

  // Constructors
  OpenCLGlobalHostAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLGlobalHostAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLGlobalHostAddressSpace; }
};

class CLANG_ABI OpenCLIntelReqdSubGroupSizeAttr : public InheritableAttr {
unsigned subGroupSize;

public:
  // Factory methods
  static OpenCLIntelReqdSubGroupSizeAttr *CreateImplicit(ASTContext &Ctx, unsigned SubGroupSize, const AttributeCommonInfo &CommonInfo);
  static OpenCLIntelReqdSubGroupSizeAttr *Create(ASTContext &Ctx, unsigned SubGroupSize, const AttributeCommonInfo &CommonInfo);
  static OpenCLIntelReqdSubGroupSizeAttr *CreateImplicit(ASTContext &Ctx, unsigned SubGroupSize, SourceRange Range = {});
  static OpenCLIntelReqdSubGroupSizeAttr *Create(ASTContext &Ctx, unsigned SubGroupSize, SourceRange Range = {});

  // Constructors
  OpenCLIntelReqdSubGroupSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned SubGroupSize
             );

  OpenCLIntelReqdSubGroupSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getSubGroupSize() const {
    return subGroupSize;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLIntelReqdSubGroupSize; }
};

class CLANG_ABI OpenCLKernelAttr : public InheritableAttr {
public:
  enum Spelling {
    Keyword_kernel = 0,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLKernelAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLKernelAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLKernelAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_kernel);
  static OpenCLKernelAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_kernel);

  // Constructors
  OpenCLKernelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLKernelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLKernel; }
};

class CLANG_ABI OpenCLLocalAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    Keyword_local = 0,
    GNU_opencl_local = 2,
    CXX11_clang_opencl_local = 3,
    C23_clang_opencl_local = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLLocalAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLLocalAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLLocalAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_local);
  static OpenCLLocalAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_local);

  // Constructors
  OpenCLLocalAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLLocalAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLLocalAddressSpace; }
};

class CLANG_ABI OpenCLPrivateAddressSpaceAttr : public TypeAttr {
public:
  enum Spelling {
    Keyword_private = 0,
    GNU_opencl_private = 2,
    CXX11_clang_opencl_private = 3,
    C23_clang_opencl_private = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OpenCLPrivateAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLPrivateAddressSpaceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OpenCLPrivateAddressSpaceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_private);
  static OpenCLPrivateAddressSpaceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Keyword_private);

  // Constructors
  OpenCLPrivateAddressSpaceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLPrivateAddressSpaceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLPrivateAddressSpace; }
};

class CLANG_ABI OpenCLUnrollHintAttr : public StmtAttr {
unsigned unrollHint;

public:
  // Factory methods
  static OpenCLUnrollHintAttr *CreateImplicit(ASTContext &Ctx, unsigned UnrollHint, const AttributeCommonInfo &CommonInfo);
  static OpenCLUnrollHintAttr *Create(ASTContext &Ctx, unsigned UnrollHint, const AttributeCommonInfo &CommonInfo);
  static OpenCLUnrollHintAttr *CreateImplicit(ASTContext &Ctx, unsigned UnrollHint, SourceRange Range = {});
  static OpenCLUnrollHintAttr *Create(ASTContext &Ctx, unsigned UnrollHint, SourceRange Range = {});

  // Constructors
  OpenCLUnrollHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned UnrollHint
             );
  OpenCLUnrollHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OpenCLUnrollHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getUnrollHint() const {
    return unrollHint;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::OpenCLUnrollHint; }
};

class CLANG_ABI OptimizeNoneAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_optnone = 0,
    CXX11_clang_optnone = 1,
    C23_clang_optnone = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OptimizeNoneAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OptimizeNoneAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OptimizeNoneAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_optnone);
  static OptimizeNoneAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_optnone);

  // Constructors
  OptimizeNoneAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OptimizeNoneAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::OptimizeNone; }
};

class CLANG_ABI OverloadableAttr : public Attr {
public:
  enum Spelling {
    GNU_overloadable = 0,
    CXX11_clang_overloadable = 1,
    C23_clang_overloadable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OverloadableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OverloadableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OverloadableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_overloadable);
  static OverloadableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_overloadable);

  // Constructors
  OverloadableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OverloadableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Overloadable; }
};

class CLANG_ABI OverrideAttr : public InheritableAttr {
public:
  // Factory methods
  static OverrideAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OverrideAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static OverrideAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static OverrideAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  OverrideAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OverrideAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Override; }
};

class CLANG_ABI OwnerAttr : public InheritableAttr {
TypeSourceInfo * derefType;

public:
  // Factory methods
  static OwnerAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * DerefType, const AttributeCommonInfo &CommonInfo);
  static OwnerAttr *Create(ASTContext &Ctx, TypeSourceInfo * DerefType, const AttributeCommonInfo &CommonInfo);
  static OwnerAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * DerefType, SourceRange Range = {});
  static OwnerAttr *Create(ASTContext &Ctx, TypeSourceInfo * DerefType, SourceRange Range = {});

  // Constructors
  OwnerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * DerefType
             );
  OwnerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  OwnerAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getDerefType() const {
    return derefType->getType();
  }  TypeSourceInfo * getDerefTypeLoc() const {
    return derefType;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Owner; }
};

class CLANG_ABI OwnershipAttr : public InheritableAttr {
IdentifierInfo * module;

  unsigned args_Size;
  ParamIdx *args_;

public:
  enum Spelling {
    GNU_ownership_holds = 0,
    CXX11_clang_ownership_holds = 1,
    C23_clang_ownership_holds = 2,
    GNU_ownership_returns = 3,
    CXX11_clang_ownership_returns = 4,
    C23_clang_ownership_returns = 5,
    GNU_ownership_takes = 6,
    CXX11_clang_ownership_takes = 7,
    C23_clang_ownership_takes = 8,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static OwnershipAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Module, ParamIdx *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static OwnershipAttr *Create(ASTContext &Ctx, IdentifierInfo * Module, ParamIdx *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static OwnershipAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * Module, ParamIdx *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_ownership_holds);
  static OwnershipAttr *Create(ASTContext &Ctx, IdentifierInfo * Module, ParamIdx *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_ownership_holds);

  // Constructors
  OwnershipAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Module
              , ParamIdx *Args, unsigned ArgsSize
             );
  OwnershipAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * Module
             );

  OwnershipAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isHolds() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1 ||
    getAttributeSpellingListIndex() == 2; }
  bool isReturns() const { return getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }
  bool isTakes() const { return getAttributeSpellingListIndex() == 6 ||
    getAttributeSpellingListIndex() == 7 ||
    getAttributeSpellingListIndex() == 8; }
  IdentifierInfo * getModule() const {
    return module;
  }

  typedef ParamIdx* args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }



    enum OwnershipKind { Holds, Returns, Takes };
    OwnershipKind getOwnKind() const {
      return isHolds() ? Holds :
             isTakes() ? Takes :
             Returns;
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::Ownership; }
};

class CLANG_ABI PackedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_packed = 0,
    CXX11_gnu_packed = 1,
    C23_gnu_packed = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PackedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PackedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PackedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_packed);
  static PackedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_packed);

  // Constructors
  PackedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PackedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Packed; }
};

class CLANG_ABI ParamTypestateAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Unknown,
    Consumed,
    Unconsumed
  };
private:
  ParamTypestateAttr::ConsumedState paramState;

public:
  enum Spelling {
    GNU_param_typestate = 0,
    CXX11_clang_param_typestate = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ParamTypestateAttr *CreateImplicit(ASTContext &Ctx, ParamTypestateAttr::ConsumedState ParamState, const AttributeCommonInfo &CommonInfo);
  static ParamTypestateAttr *Create(ASTContext &Ctx, ParamTypestateAttr::ConsumedState ParamState, const AttributeCommonInfo &CommonInfo);
  static ParamTypestateAttr *CreateImplicit(ASTContext &Ctx, ParamTypestateAttr::ConsumedState ParamState, SourceRange Range = {}, Spelling S = GNU_param_typestate);
  static ParamTypestateAttr *Create(ASTContext &Ctx, ParamTypestateAttr::ConsumedState ParamState, SourceRange Range = {}, Spelling S = GNU_param_typestate);

  // Constructors
  ParamTypestateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ParamTypestateAttr::ConsumedState ParamState
             );

  ParamTypestateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ParamTypestateAttr::ConsumedState getParamState() const {
    return paramState;
  }

  static bool ConvertStrToConsumedState(StringRef Val, ParamTypestateAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(ParamTypestateAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ParamTypestate; }
};

class CLANG_ABI PascalAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_pascal = 0,
    CXX11_clang_pascal = 1,
    C23_clang_pascal = 2,
    Keyword_pascal = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PascalAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PascalAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PascalAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pascal);
  static PascalAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pascal);

  // Constructors
  PascalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PascalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Pascal; }
};

class CLANG_ABI PassObjectSizeAttr : public InheritableParamAttr {
int type;

public:
  enum Spelling {
    GNU_pass_object_size = 0,
    CXX11_clang_pass_object_size = 1,
    C23_clang_pass_object_size = 2,
    GNU_pass_dynamic_object_size = 3,
    CXX11_clang_pass_dynamic_object_size = 4,
    C23_clang_pass_dynamic_object_size = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PassObjectSizeAttr *CreateImplicit(ASTContext &Ctx, int Type, const AttributeCommonInfo &CommonInfo);
  static PassObjectSizeAttr *Create(ASTContext &Ctx, int Type, const AttributeCommonInfo &CommonInfo);
  static PassObjectSizeAttr *CreateImplicit(ASTContext &Ctx, int Type, SourceRange Range = {}, Spelling S = GNU_pass_object_size);
  static PassObjectSizeAttr *Create(ASTContext &Ctx, int Type, SourceRange Range = {}, Spelling S = GNU_pass_object_size);

  // Constructors
  PassObjectSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Type
             );

  PassObjectSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isDynamic() const { return getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }
  int getType() const {
    return type;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PassObjectSize; }
};

class CLANG_ABI PatchableFunctionEntryAttr : public InheritableAttr {
unsigned count;

int offset;

unsigned sectionLength;
char *section;

public:
  enum Spelling {
    GNU_patchable_function_entry = 0,
    CXX11_gnu_patchable_function_entry = 1,
    C23_gnu_patchable_function_entry = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PatchableFunctionEntryAttr *CreateImplicit(ASTContext &Ctx, unsigned Count, int Offset, llvm::StringRef Section, const AttributeCommonInfo &CommonInfo);
  static PatchableFunctionEntryAttr *Create(ASTContext &Ctx, unsigned Count, int Offset, llvm::StringRef Section, const AttributeCommonInfo &CommonInfo);
  static PatchableFunctionEntryAttr *CreateImplicit(ASTContext &Ctx, unsigned Count, int Offset, llvm::StringRef Section, SourceRange Range = {}, Spelling S = GNU_patchable_function_entry);
  static PatchableFunctionEntryAttr *Create(ASTContext &Ctx, unsigned Count, int Offset, llvm::StringRef Section, SourceRange Range = {}, Spelling S = GNU_patchable_function_entry);

  // Constructors
  PatchableFunctionEntryAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Count
              , int Offset
              , llvm::StringRef Section
             );
  PatchableFunctionEntryAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned Count
             );

  PatchableFunctionEntryAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getCount() const {
    return count;
  }

  int getOffset() const {
    return offset;
  }

  static const int DefaultOffset = 0;

  llvm::StringRef getSection() const {
    return llvm::StringRef(section, sectionLength);
  }
  unsigned getSectionLength() const {
    return sectionLength;
  }
  void setSection(ASTContext &C, llvm::StringRef S) {
    sectionLength = S.size();
    this->section = new (C, 1) char [sectionLength];
    if (!S.empty())
      std::memcpy(this->section, S.data(), sectionLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PatchableFunctionEntry; }
};

class CLANG_ABI PcsAttr : public InheritableAttr {
public:
  enum PCSType {
    AAPCS,
    AAPCS_VFP
  };
private:
  PcsAttr::PCSType pCS;

public:
  enum Spelling {
    GNU_pcs = 0,
    CXX11_gnu_pcs = 1,
    C23_gnu_pcs = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PcsAttr *CreateImplicit(ASTContext &Ctx, PcsAttr::PCSType PCS, const AttributeCommonInfo &CommonInfo);
  static PcsAttr *Create(ASTContext &Ctx, PcsAttr::PCSType PCS, const AttributeCommonInfo &CommonInfo);
  static PcsAttr *CreateImplicit(ASTContext &Ctx, PcsAttr::PCSType PCS, SourceRange Range = {}, Spelling S = GNU_pcs);
  static PcsAttr *Create(ASTContext &Ctx, PcsAttr::PCSType PCS, SourceRange Range = {}, Spelling S = GNU_pcs);

  // Constructors
  PcsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , PcsAttr::PCSType PCS
             );

  PcsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  PcsAttr::PCSType getPCS() const {
    return pCS;
  }

  static bool ConvertStrToPCSType(StringRef Val, PcsAttr::PCSType &Out);
  static const char *ConvertPCSTypeToStr(PcsAttr::PCSType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::Pcs; }
};

class CLANG_ABI PointerAttr : public InheritableAttr {
TypeSourceInfo * derefType;

public:
  // Factory methods
  static PointerAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * DerefType, const AttributeCommonInfo &CommonInfo);
  static PointerAttr *Create(ASTContext &Ctx, TypeSourceInfo * DerefType, const AttributeCommonInfo &CommonInfo);
  static PointerAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * DerefType, SourceRange Range = {});
  static PointerAttr *Create(ASTContext &Ctx, TypeSourceInfo * DerefType, SourceRange Range = {});

  // Constructors
  PointerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * DerefType
             );
  PointerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PointerAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getDerefType() const {
    return derefType->getType();
  }  TypeSourceInfo * getDerefTypeLoc() const {
    return derefType;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Pointer; }
};

class CLANG_ABI PointerAuthAttr : public TypeAttr {
int key;

bool addressDiscriminated;

int extraDiscriminator;

public:
  // Factory methods
  static PointerAuthAttr *CreateImplicit(ASTContext &Ctx, int Key, bool AddressDiscriminated, int ExtraDiscriminator, const AttributeCommonInfo &CommonInfo);
  static PointerAuthAttr *Create(ASTContext &Ctx, int Key, bool AddressDiscriminated, int ExtraDiscriminator, const AttributeCommonInfo &CommonInfo);
  static PointerAuthAttr *CreateImplicit(ASTContext &Ctx, int Key, bool AddressDiscriminated, int ExtraDiscriminator, SourceRange Range = {});
  static PointerAuthAttr *Create(ASTContext &Ctx, int Key, bool AddressDiscriminated, int ExtraDiscriminator, SourceRange Range = {});

  // Constructors
  PointerAuthAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Key
              , bool AddressDiscriminated
              , int ExtraDiscriminator
             );
  PointerAuthAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Key
             );

  PointerAuthAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getKey() const {
    return key;
  }

  bool getAddressDiscriminated() const {
    return addressDiscriminated;
  }

  int getExtraDiscriminator() const {
    return extraDiscriminator;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PointerAuth; }
};

class CLANG_ABI PragmaClangBSSSectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static PragmaClangBSSSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangBSSSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangBSSSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static PragmaClangBSSSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  PragmaClangBSSSectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  PragmaClangBSSSectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PragmaClangBSSSection; }
};

class CLANG_ABI PragmaClangDataSectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static PragmaClangDataSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangDataSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangDataSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static PragmaClangDataSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  PragmaClangDataSectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  PragmaClangDataSectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PragmaClangDataSection; }
};

class CLANG_ABI PragmaClangRelroSectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static PragmaClangRelroSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangRelroSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangRelroSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static PragmaClangRelroSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  PragmaClangRelroSectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  PragmaClangRelroSectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PragmaClangRelroSection; }
};

class CLANG_ABI PragmaClangRodataSectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static PragmaClangRodataSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangRodataSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangRodataSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static PragmaClangRodataSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  PragmaClangRodataSectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  PragmaClangRodataSectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PragmaClangRodataSection; }
};

class CLANG_ABI PragmaClangTextSectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static PragmaClangTextSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangTextSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static PragmaClangTextSectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static PragmaClangTextSectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  PragmaClangTextSectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  PragmaClangTextSectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PragmaClangTextSection; }
};

class CLANG_ABI PreferredNameAttr : public InheritableAttr {
TypeSourceInfo * typedefType;

public:
  enum Spelling {
    GNU_preferred_name = 0,
    CXX11_clang_preferred_name = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PreferredNameAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * TypedefType, const AttributeCommonInfo &CommonInfo);
  static PreferredNameAttr *Create(ASTContext &Ctx, TypeSourceInfo * TypedefType, const AttributeCommonInfo &CommonInfo);
  static PreferredNameAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * TypedefType, SourceRange Range = {}, Spelling S = GNU_preferred_name);
  static PreferredNameAttr *Create(ASTContext &Ctx, TypeSourceInfo * TypedefType, SourceRange Range = {}, Spelling S = GNU_preferred_name);

  // Constructors
  PreferredNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * TypedefType
             );

  PreferredNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getTypedefType() const {
    return typedefType->getType();
  }  TypeSourceInfo * getTypedefTypeLoc() const {
    return typedefType;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PreferredName; }
};

class CLANG_ABI PreferredTypeAttr : public InheritableAttr {
TypeSourceInfo * type;

public:
  enum Spelling {
    GNU_preferred_type = 0,
    CXX11_clang_preferred_type = 1,
    C23_clang_preferred_type = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PreferredTypeAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Type, const AttributeCommonInfo &CommonInfo);
  static PreferredTypeAttr *Create(ASTContext &Ctx, TypeSourceInfo * Type, const AttributeCommonInfo &CommonInfo);
  static PreferredTypeAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * Type, SourceRange Range = {}, Spelling S = GNU_preferred_type);
  static PreferredTypeAttr *Create(ASTContext &Ctx, TypeSourceInfo * Type, SourceRange Range = {}, Spelling S = GNU_preferred_type);

  // Constructors
  PreferredTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * Type
             );
  PreferredTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PreferredTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getType() const {
    return type->getType();
  }  TypeSourceInfo * getTypeLoc() const {
    return type;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PreferredType; }
};

class CLANG_ABI PreserveAllAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_preserve_all = 0,
    CXX11_clang_preserve_all = 1,
    C23_clang_preserve_all = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PreserveAllAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveAllAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveAllAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_all);
  static PreserveAllAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_all);

  // Constructors
  PreserveAllAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PreserveAllAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::PreserveAll; }
};

class CLANG_ABI PreserveMostAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_preserve_most = 0,
    CXX11_clang_preserve_most = 1,
    C23_clang_preserve_most = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PreserveMostAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveMostAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveMostAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_most);
  static PreserveMostAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_most);

  // Constructors
  PreserveMostAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PreserveMostAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::PreserveMost; }
};

class CLANG_ABI PreserveNoneAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_preserve_none = 0,
    CXX11_clang_preserve_none = 1,
    C23_clang_preserve_none = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PreserveNoneAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveNoneAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PreserveNoneAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_none);
  static PreserveNoneAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_preserve_none);

  // Constructors
  PreserveNoneAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PreserveNoneAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::PreserveNone; }
};

class CLANG_ABI PtGuardedByAttr : public InheritableAttr {
Expr * arg;

public:
  // Factory methods
  static PtGuardedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static PtGuardedByAttr *Create(ASTContext &Ctx, Expr * Arg, const AttributeCommonInfo &CommonInfo);
  static PtGuardedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});
  static PtGuardedByAttr *Create(ASTContext &Ctx, Expr * Arg, SourceRange Range = {});

  // Constructors
  PtGuardedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Arg
             );

  PtGuardedByAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getArg() const {
    return arg;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::PtGuardedBy; }
};

class CLANG_ABI PtGuardedVarAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_pt_guarded_var = 0,
    CXX11_clang_pt_guarded_var = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PtGuardedVarAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PtGuardedVarAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PtGuardedVarAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pt_guarded_var);
  static PtGuardedVarAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pt_guarded_var);

  // Constructors
  PtGuardedVarAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PtGuardedVarAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::PtGuardedVar; }
};

class CLANG_ABI Ptr32Attr : public TypeAttr {
public:
  // Factory methods
  static Ptr32Attr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Ptr32Attr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Ptr32Attr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static Ptr32Attr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  Ptr32Attr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  Ptr32Attr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Ptr32; }
};

class CLANG_ABI Ptr64Attr : public TypeAttr {
public:
  // Factory methods
  static Ptr64Attr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Ptr64Attr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static Ptr64Attr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static Ptr64Attr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  Ptr64Attr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  Ptr64Attr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Ptr64; }
};

class CLANG_ABI PureAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_pure = 0,
    CXX11_gnu_pure = 1,
    C23_gnu_pure = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static PureAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PureAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static PureAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pure);
  static PureAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_pure);

  // Constructors
  PureAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  PureAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Pure; }
};

class CLANG_ABI RISCVInterruptAttr : public InheritableAttr {
public:
  enum InterruptType {
    supervisor,
    machine,
    qcinest,
    qcinonest
  };
private:
  RISCVInterruptAttr::InterruptType interrupt;

public:
  enum Spelling {
    GNU_interrupt = 0,
    CXX11_gnu_interrupt = 1,
    C23_gnu_interrupt = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RISCVInterruptAttr *CreateImplicit(ASTContext &Ctx, RISCVInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static RISCVInterruptAttr *Create(ASTContext &Ctx, RISCVInterruptAttr::InterruptType Interrupt, const AttributeCommonInfo &CommonInfo);
  static RISCVInterruptAttr *CreateImplicit(ASTContext &Ctx, RISCVInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);
  static RISCVInterruptAttr *Create(ASTContext &Ctx, RISCVInterruptAttr::InterruptType Interrupt, SourceRange Range = {}, Spelling S = GNU_interrupt);

  // Constructors
  RISCVInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , RISCVInterruptAttr::InterruptType Interrupt
             );
  RISCVInterruptAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RISCVInterruptAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  RISCVInterruptAttr::InterruptType getInterrupt() const {
    return interrupt;
  }

  static bool ConvertStrToInterruptType(StringRef Val, RISCVInterruptAttr::InterruptType &Out);
  static const char *ConvertInterruptTypeToStr(RISCVInterruptAttr::InterruptType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::RISCVInterrupt; }
};

class CLANG_ABI RISCVVLSCCAttr : public InheritableAttr {
unsigned vectorWidth;

public:
  enum Spelling {
    CXX11_riscv_vls_cc = 0,
    C23_riscv_vls_cc = 1,
    GNU_riscv_vls_cc = 2,
    CXX11_clang_riscv_vls_cc = 3,
    C23_clang_riscv_vls_cc = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RISCVVLSCCAttr *CreateImplicit(ASTContext &Ctx, unsigned VectorWidth, const AttributeCommonInfo &CommonInfo);
  static RISCVVLSCCAttr *Create(ASTContext &Ctx, unsigned VectorWidth, const AttributeCommonInfo &CommonInfo);
  static RISCVVLSCCAttr *CreateImplicit(ASTContext &Ctx, unsigned VectorWidth, SourceRange Range = {}, Spelling S = CXX11_riscv_vls_cc);
  static RISCVVLSCCAttr *Create(ASTContext &Ctx, unsigned VectorWidth, SourceRange Range = {}, Spelling S = CXX11_riscv_vls_cc);

  // Constructors
  RISCVVLSCCAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned VectorWidth
             );
  RISCVVLSCCAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RISCVVLSCCAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  unsigned getVectorWidth() const {
    return vectorWidth;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::RISCVVLSCC; }
};

class CLANG_ABI RISCVVectorCCAttr : public InheritableAttr {
public:
  enum Spelling {
    CXX11_riscv_vector_cc = 0,
    C23_riscv_vector_cc = 1,
    GNU_riscv_vector_cc = 2,
    CXX11_clang_riscv_vector_cc = 3,
    C23_clang_riscv_vector_cc = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RISCVVectorCCAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RISCVVectorCCAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RISCVVectorCCAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_riscv_vector_cc);
  static RISCVVectorCCAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_riscv_vector_cc);

  // Constructors
  RISCVVectorCCAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RISCVVectorCCAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::RISCVVectorCC; }
};

class CLANG_ABI RandomizeLayoutAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_randomize_layout = 0,
    CXX11_gnu_randomize_layout = 1,
    C23_gnu_randomize_layout = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RandomizeLayoutAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RandomizeLayoutAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RandomizeLayoutAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_randomize_layout);
  static RandomizeLayoutAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_randomize_layout);

  // Constructors
  RandomizeLayoutAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RandomizeLayoutAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::RandomizeLayout; }
};

class CLANG_ABI ReadOnlyPlacementAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_enforce_read_only_placement = 0,
    CXX11_clang_enforce_read_only_placement = 1,
    C23_clang_enforce_read_only_placement = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReadOnlyPlacementAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReadOnlyPlacementAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReadOnlyPlacementAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_enforce_read_only_placement);
  static ReadOnlyPlacementAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_enforce_read_only_placement);

  // Constructors
  ReadOnlyPlacementAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ReadOnlyPlacementAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ReadOnlyPlacement; }
};

class CLANG_ABI RegCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_regcall = 0,
    CXX11_gnu_regcall = 1,
    C23_gnu_regcall = 2,
    Keyword_regcall = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RegCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RegCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RegCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_regcall);
  static RegCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_regcall);

  // Constructors
  RegCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RegCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::RegCall; }
};

class CLANG_ABI ReinitializesAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_reinitializes = 0,
    CXX11_clang_reinitializes = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReinitializesAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReinitializesAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReinitializesAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_reinitializes);
  static ReinitializesAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_reinitializes);

  // Constructors
  ReinitializesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ReinitializesAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Reinitializes; }
};

class CLANG_ABI ReleaseCapabilityAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  enum Spelling {
    GNU_release_capability = 0,
    CXX11_clang_release_capability = 1,
    GNU_release_shared_capability = 2,
    CXX11_clang_release_shared_capability = 3,
    GNU_release_generic_capability = 4,
    CXX11_clang_release_generic_capability = 5,
    GNU_unlock_function = 6,
    CXX11_clang_unlock_function = 7,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReleaseCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static ReleaseCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static ReleaseCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_release_capability);
  static ReleaseCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_release_capability);

  // Constructors
  ReleaseCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  ReleaseCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ReleaseCapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3; }
  bool isGeneric() const { return getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5 ||
    getAttributeSpellingListIndex() == 6 ||
    getAttributeSpellingListIndex() == 7; }
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::ReleaseCapability; }
};

class CLANG_ABI ReleaseHandleAttr : public InheritableParamAttr {
unsigned handleTypeLength;
char *handleType;

public:
  enum Spelling {
    GNU_release_handle = 0,
    CXX11_clang_release_handle = 1,
    C23_clang_release_handle = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReleaseHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static ReleaseHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static ReleaseHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_release_handle);
  static ReleaseHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_release_handle);

  // Constructors
  ReleaseHandleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef HandleType
             );

  ReleaseHandleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getHandleType() const {
    return llvm::StringRef(handleType, handleTypeLength);
  }
  unsigned getHandleTypeLength() const {
    return handleTypeLength;
  }
  void setHandleType(ASTContext &C, llvm::StringRef S) {
    handleTypeLength = S.size();
    this->handleType = new (C, 1) char [handleTypeLength];
    if (!S.empty())
      std::memcpy(this->handleType, S.data(), handleTypeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ReleaseHandle; }
};

class CLANG_ABI ReqdWorkGroupSizeAttr : public InheritableAttr {
Expr * xDim;

Expr * yDim;

Expr * zDim;

public:
  // Factory methods
  static ReqdWorkGroupSizeAttr *CreateImplicit(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, const AttributeCommonInfo &CommonInfo);
  static ReqdWorkGroupSizeAttr *Create(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, const AttributeCommonInfo &CommonInfo);
  static ReqdWorkGroupSizeAttr *CreateImplicit(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, SourceRange Range = {});
  static ReqdWorkGroupSizeAttr *Create(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, SourceRange Range = {});

  // Constructors
  ReqdWorkGroupSizeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * XDim
              , Expr * YDim
              , Expr * ZDim
             );

  ReqdWorkGroupSizeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getXDim() const {
    return xDim;
  }

  Expr * getYDim() const {
    return yDim;
  }

  Expr * getZDim() const {
    return zDim;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::ReqdWorkGroupSize; }
};

class CLANG_ABI RequiresCapabilityAttr : public InheritableAttr {
  unsigned args_Size;
  Expr * *args_;

public:
  enum Spelling {
    GNU_requires_capability = 0,
    CXX11_clang_requires_capability = 1,
    GNU_exclusive_locks_required = 2,
    CXX11_clang_exclusive_locks_required = 3,
    GNU_requires_shared_capability = 4,
    CXX11_clang_requires_shared_capability = 5,
    GNU_shared_locks_required = 6,
    CXX11_clang_shared_locks_required = 7,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RequiresCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static RequiresCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static RequiresCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_requires_capability);
  static RequiresCapabilityAttr *Create(ASTContext &Ctx, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_requires_capability);

  // Constructors
  RequiresCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * *Args, unsigned ArgsSize
             );
  RequiresCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RequiresCapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5 ||
    getAttributeSpellingListIndex() == 6 ||
    getAttributeSpellingListIndex() == 7; }
  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::RequiresCapability; }
};

class CLANG_ABI RestrictAttr : public InheritableAttr {
Expr * deallocator;

ParamIdx deallocatorPtrArgIndex;

public:
  enum Spelling {
    Declspec_restrict = 0,
    GNU_malloc = 1,
    CXX11_gnu_malloc = 2,
    C23_gnu_malloc = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RestrictAttr *CreateImplicit(ASTContext &Ctx, Expr * Deallocator, ParamIdx DeallocatorPtrArgIndex, const AttributeCommonInfo &CommonInfo);
  static RestrictAttr *Create(ASTContext &Ctx, Expr * Deallocator, ParamIdx DeallocatorPtrArgIndex, const AttributeCommonInfo &CommonInfo);
  static RestrictAttr *CreateImplicit(ASTContext &Ctx, Expr * Deallocator, ParamIdx DeallocatorPtrArgIndex, SourceRange Range = {}, Spelling S = Declspec_restrict);
  static RestrictAttr *Create(ASTContext &Ctx, Expr * Deallocator, ParamIdx DeallocatorPtrArgIndex, SourceRange Range = {}, Spelling S = Declspec_restrict);

  // Constructors
  RestrictAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Deallocator
              , ParamIdx DeallocatorPtrArgIndex
             );
  RestrictAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RestrictAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  Expr * getDeallocator() const {
    return deallocator;
  }

  ParamIdx getDeallocatorPtrArgIndex() const {
    return deallocatorPtrArgIndex;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Restrict; }
};

class CLANG_ABI RetainAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_retain = 0,
    CXX11_gnu_retain = 1,
    C23_gnu_retain = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static RetainAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RetainAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static RetainAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_retain);
  static RetainAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_retain);

  // Constructors
  RetainAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  RetainAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Retain; }
};

class CLANG_ABI ReturnTypestateAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Unknown,
    Consumed,
    Unconsumed
  };
private:
  ReturnTypestateAttr::ConsumedState state;

public:
  enum Spelling {
    GNU_return_typestate = 0,
    CXX11_clang_return_typestate = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReturnTypestateAttr *CreateImplicit(ASTContext &Ctx, ReturnTypestateAttr::ConsumedState State, const AttributeCommonInfo &CommonInfo);
  static ReturnTypestateAttr *Create(ASTContext &Ctx, ReturnTypestateAttr::ConsumedState State, const AttributeCommonInfo &CommonInfo);
  static ReturnTypestateAttr *CreateImplicit(ASTContext &Ctx, ReturnTypestateAttr::ConsumedState State, SourceRange Range = {}, Spelling S = GNU_return_typestate);
  static ReturnTypestateAttr *Create(ASTContext &Ctx, ReturnTypestateAttr::ConsumedState State, SourceRange Range = {}, Spelling S = GNU_return_typestate);

  // Constructors
  ReturnTypestateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ReturnTypestateAttr::ConsumedState State
             );

  ReturnTypestateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ReturnTypestateAttr::ConsumedState getState() const {
    return state;
  }

  static bool ConvertStrToConsumedState(StringRef Val, ReturnTypestateAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(ReturnTypestateAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ReturnTypestate; }
};

class CLANG_ABI ReturnsNonNullAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_returns_nonnull = 0,
    CXX11_gnu_returns_nonnull = 1,
    C23_gnu_returns_nonnull = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReturnsNonNullAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReturnsNonNullAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReturnsNonNullAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_returns_nonnull);
  static ReturnsNonNullAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_returns_nonnull);

  // Constructors
  ReturnsNonNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ReturnsNonNullAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ReturnsNonNull; }
};

class CLANG_ABI ReturnsTwiceAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_returns_twice = 0,
    CXX11_gnu_returns_twice = 1,
    C23_gnu_returns_twice = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ReturnsTwiceAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReturnsTwiceAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ReturnsTwiceAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_returns_twice);
  static ReturnsTwiceAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_returns_twice);

  // Constructors
  ReturnsTwiceAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ReturnsTwiceAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ReturnsTwice; }
};

class CLANG_ABI SPtrAttr : public TypeAttr {
public:
  // Factory methods
  static SPtrAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SPtrAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SPtrAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SPtrAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SPtrAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SPtrAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SPtr; }
};

class CLANG_ABI SYCLKernelAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_sycl_kernel = 0,
    CXX11_clang_sycl_kernel = 1,
    C23_clang_sycl_kernel = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SYCLKernelAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sycl_kernel);
  static SYCLKernelAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sycl_kernel);

  // Constructors
  SYCLKernelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SYCLKernelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SYCLKernel; }
};

class CLANG_ABI SYCLKernelEntryPointAttr : public InheritableAttr {
TypeSourceInfo * kernelName;

bool invalid;

public:
  enum Spelling {
    GNU_sycl_kernel_entry_point = 0,
    CXX11_clang_sycl_kernel_entry_point = 1,
    C23_clang_sycl_kernel_entry_point = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SYCLKernelEntryPointAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * KernelName, bool Invalid, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelEntryPointAttr *Create(ASTContext &Ctx, TypeSourceInfo * KernelName, bool Invalid, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelEntryPointAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * KernelName, bool Invalid, SourceRange Range = {}, Spelling S = GNU_sycl_kernel_entry_point);
  static SYCLKernelEntryPointAttr *Create(ASTContext &Ctx, TypeSourceInfo * KernelName, bool Invalid, SourceRange Range = {}, Spelling S = GNU_sycl_kernel_entry_point);
  static SYCLKernelEntryPointAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * KernelName, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelEntryPointAttr *Create(ASTContext &Ctx, TypeSourceInfo * KernelName, const AttributeCommonInfo &CommonInfo);
  static SYCLKernelEntryPointAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * KernelName, SourceRange Range = {}, Spelling S = GNU_sycl_kernel_entry_point);
  static SYCLKernelEntryPointAttr *Create(ASTContext &Ctx, TypeSourceInfo * KernelName, SourceRange Range = {}, Spelling S = GNU_sycl_kernel_entry_point);

  // Constructors
  SYCLKernelEntryPointAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * KernelName
              , bool Invalid
             );
  SYCLKernelEntryPointAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * KernelName
             );

  SYCLKernelEntryPointAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getKernelName() const {
    return kernelName->getType();
  }  TypeSourceInfo * getKernelNameLoc() const {
    return kernelName;
  }

  bool getInvalid() const {
    return invalid;
  }

  static const bool DefaultInvalid = false;


    void setInvalidAttr() { invalid = true; }
    bool isInvalidAttr() const { return invalid; }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::SYCLKernelEntryPoint; }
};

class CLANG_ABI SYCLSpecialClassAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_sycl_special_class = 0,
    CXX11_clang_sycl_special_class = 1,
    C23_clang_sycl_special_class = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SYCLSpecialClassAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SYCLSpecialClassAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SYCLSpecialClassAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sycl_special_class);
  static SYCLSpecialClassAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sycl_special_class);

  // Constructors
  SYCLSpecialClassAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SYCLSpecialClassAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SYCLSpecialClass; }
};

class CLANG_ABI ScopedLockableAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_scoped_lockable = 0,
    CXX11_clang_scoped_lockable = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ScopedLockableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ScopedLockableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ScopedLockableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_scoped_lockable);
  static ScopedLockableAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_scoped_lockable);

  // Constructors
  ScopedLockableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ScopedLockableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ScopedLockable; }
};

class CLANG_ABI SectionAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  enum Spelling {
    GNU_section = 0,
    CXX11_gnu_section = 1,
    C23_gnu_section = 2,
    Declspec_allocate = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SectionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {}, Spelling S = GNU_section);
  static SectionAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {}, Spelling S = GNU_section);

  // Constructors
  SectionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  SectionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Section; }
};

class CLANG_ABI SelectAnyAttr : public InheritableAttr {
public:
  enum Spelling {
    Declspec_selectany = 0,
    GNU_selectany = 1,
    CXX11_gnu_selectany = 2,
    C23_gnu_selectany = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SelectAnyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SelectAnyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SelectAnyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_selectany);
  static SelectAnyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = Declspec_selectany);

  // Constructors
  SelectAnyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SelectAnyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SelectAny; }
};

class CLANG_ABI SentinelAttr : public InheritableAttr {
int sentinel;

int nullPos;

public:
  enum Spelling {
    GNU_sentinel = 0,
    CXX11_gnu_sentinel = 1,
    C23_gnu_sentinel = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SentinelAttr *CreateImplicit(ASTContext &Ctx, int Sentinel, int NullPos, const AttributeCommonInfo &CommonInfo);
  static SentinelAttr *Create(ASTContext &Ctx, int Sentinel, int NullPos, const AttributeCommonInfo &CommonInfo);
  static SentinelAttr *CreateImplicit(ASTContext &Ctx, int Sentinel, int NullPos, SourceRange Range = {}, Spelling S = GNU_sentinel);
  static SentinelAttr *Create(ASTContext &Ctx, int Sentinel, int NullPos, SourceRange Range = {}, Spelling S = GNU_sentinel);

  // Constructors
  SentinelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , int Sentinel
              , int NullPos
             );
  SentinelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SentinelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  int getSentinel() const {
    return sentinel;
  }

  static const int DefaultSentinel = 0;

  int getNullPos() const {
    return nullPos;
  }

  static const int DefaultNullPos = 0;



  static bool classof(const Attr *A) { return A->getKind() == attr::Sentinel; }
};

class CLANG_ABI SetTypestateAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Unknown,
    Consumed,
    Unconsumed
  };
private:
  SetTypestateAttr::ConsumedState newState;

public:
  enum Spelling {
    GNU_set_typestate = 0,
    CXX11_clang_set_typestate = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SetTypestateAttr *CreateImplicit(ASTContext &Ctx, SetTypestateAttr::ConsumedState NewState, const AttributeCommonInfo &CommonInfo);
  static SetTypestateAttr *Create(ASTContext &Ctx, SetTypestateAttr::ConsumedState NewState, const AttributeCommonInfo &CommonInfo);
  static SetTypestateAttr *CreateImplicit(ASTContext &Ctx, SetTypestateAttr::ConsumedState NewState, SourceRange Range = {}, Spelling S = GNU_set_typestate);
  static SetTypestateAttr *Create(ASTContext &Ctx, SetTypestateAttr::ConsumedState NewState, SourceRange Range = {}, Spelling S = GNU_set_typestate);

  // Constructors
  SetTypestateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SetTypestateAttr::ConsumedState NewState
             );

  SetTypestateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  SetTypestateAttr::ConsumedState getNewState() const {
    return newState;
  }

  static bool ConvertStrToConsumedState(StringRef Val, SetTypestateAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(SetTypestateAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::SetTypestate; }
};

class CLANG_ABI SizedByAttr : public InheritableAttr {
Expr * size;

int nestedLevel;

public:
  enum Spelling {
    GNU_sized_by = 0,
    CXX11_clang_sized_by = 1,
    C23_clang_sized_by = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SizedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Size, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static SizedByAttr *Create(ASTContext &Ctx, Expr * Size, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static SizedByAttr *CreateImplicit(ASTContext &Ctx, Expr * Size, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_sized_by);
  static SizedByAttr *Create(ASTContext &Ctx, Expr * Size, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_sized_by);

  // Constructors
  SizedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Size
              , int NestedLevel
             );
  SizedByAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Size
             );

  SizedByAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getSize() const {
    return size;
  }

  int getNestedLevel() const {
    return nestedLevel;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SizedBy; }
};

class CLANG_ABI SizedByOrNullAttr : public InheritableAttr {
Expr * size;

int nestedLevel;

public:
  enum Spelling {
    GNU_sized_by_or_null = 0,
    CXX11_clang_sized_by_or_null = 1,
    C23_clang_sized_by_or_null = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SizedByOrNullAttr *CreateImplicit(ASTContext &Ctx, Expr * Size, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static SizedByOrNullAttr *Create(ASTContext &Ctx, Expr * Size, int NestedLevel, const AttributeCommonInfo &CommonInfo);
  static SizedByOrNullAttr *CreateImplicit(ASTContext &Ctx, Expr * Size, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_sized_by_or_null);
  static SizedByOrNullAttr *Create(ASTContext &Ctx, Expr * Size, int NestedLevel, SourceRange Range = {}, Spelling S = GNU_sized_by_or_null);

  // Constructors
  SizedByOrNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Size
              , int NestedLevel
             );
  SizedByOrNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * Size
             );

  SizedByOrNullAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getSize() const {
    return size;
  }

  int getNestedLevel() const {
    return nestedLevel;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SizedByOrNull; }
};

class CLANG_ABI SpeculativeLoadHardeningAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_speculative_load_hardening = 0,
    CXX11_clang_speculative_load_hardening = 1,
    C23_clang_speculative_load_hardening = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SpeculativeLoadHardeningAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SpeculativeLoadHardeningAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SpeculativeLoadHardeningAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_speculative_load_hardening);
  static SpeculativeLoadHardeningAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_speculative_load_hardening);

  // Constructors
  SpeculativeLoadHardeningAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SpeculativeLoadHardeningAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SpeculativeLoadHardening; }
};

class CLANG_ABI StandaloneDebugAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_standalone_debug = 0,
    CXX11_clang_standalone_debug = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static StandaloneDebugAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StandaloneDebugAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StandaloneDebugAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_standalone_debug);
  static StandaloneDebugAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_standalone_debug);

  // Constructors
  StandaloneDebugAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  StandaloneDebugAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::StandaloneDebug; }
};

class CLANG_ABI StdCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_stdcall = 0,
    CXX11_gnu_stdcall = 1,
    C23_gnu_stdcall = 2,
    Keyword_stdcall = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static StdCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StdCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StdCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_stdcall);
  static StdCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_stdcall);

  // Constructors
  StdCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  StdCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::StdCall; }
};

class CLANG_ABI StrictFPAttr : public InheritableAttr {
public:
  // Factory methods
  static StrictFPAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StrictFPAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StrictFPAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static StrictFPAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  StrictFPAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  StrictFPAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::StrictFP; }
};

class CLANG_ABI StrictGuardStackCheckAttr : public InheritableAttr {
public:
  // Factory methods
  static StrictGuardStackCheckAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StrictGuardStackCheckAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static StrictGuardStackCheckAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static StrictGuardStackCheckAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  StrictGuardStackCheckAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  StrictGuardStackCheckAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::StrictGuardStackCheck; }
};

class CLANG_ABI SuppressAttr : public DeclOrStmtAttr {
  unsigned diagnosticIdentifiers_Size;
  StringRef *diagnosticIdentifiers_;

public:
  enum Spelling {
    CXX11_gsl_suppress = 0,
    GNU_suppress = 1,
    CXX11_clang_suppress = 2,
    C23_clang_suppress = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SuppressAttr *CreateImplicit(ASTContext &Ctx, StringRef *DiagnosticIdentifiers, unsigned DiagnosticIdentifiersSize, const AttributeCommonInfo &CommonInfo);
  static SuppressAttr *Create(ASTContext &Ctx, StringRef *DiagnosticIdentifiers, unsigned DiagnosticIdentifiersSize, const AttributeCommonInfo &CommonInfo);
  static SuppressAttr *CreateImplicit(ASTContext &Ctx, StringRef *DiagnosticIdentifiers, unsigned DiagnosticIdentifiersSize, SourceRange Range = {}, Spelling S = CXX11_gsl_suppress);
  static SuppressAttr *Create(ASTContext &Ctx, StringRef *DiagnosticIdentifiers, unsigned DiagnosticIdentifiersSize, SourceRange Range = {}, Spelling S = CXX11_gsl_suppress);

  // Constructors
  SuppressAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *DiagnosticIdentifiers, unsigned DiagnosticIdentifiersSize
             );
  SuppressAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SuppressAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  bool isGSL() const { return getAttributeSpellingListIndex() == 0; }
  typedef StringRef* diagnosticIdentifiers_iterator;
  diagnosticIdentifiers_iterator diagnosticIdentifiers_begin() const { return diagnosticIdentifiers_; }
  diagnosticIdentifiers_iterator diagnosticIdentifiers_end() const { return diagnosticIdentifiers_ + diagnosticIdentifiers_Size; }
  unsigned diagnosticIdentifiers_size() const { return diagnosticIdentifiers_Size; }
  llvm::iterator_range<diagnosticIdentifiers_iterator> diagnosticIdentifiers() const { return llvm::make_range(diagnosticIdentifiers_begin(), diagnosticIdentifiers_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::Suppress; }
};

class CLANG_ABI SwiftAsyncAttr : public InheritableAttr {
public:
  enum Kind {
    None,
    SwiftPrivate,
    NotSwiftPrivate
  };
private:
  SwiftAsyncAttr::Kind kind;

ParamIdx completionHandlerIndex;

public:
  enum Spelling {
    GNU_swift_async = 0,
    CXX11_clang_swift_async = 1,
    C23_clang_swift_async = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftAsyncAttr *CreateImplicit(ASTContext &Ctx, SwiftAsyncAttr::Kind Kind, ParamIdx CompletionHandlerIndex, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncAttr *Create(ASTContext &Ctx, SwiftAsyncAttr::Kind Kind, ParamIdx CompletionHandlerIndex, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncAttr *CreateImplicit(ASTContext &Ctx, SwiftAsyncAttr::Kind Kind, ParamIdx CompletionHandlerIndex, SourceRange Range = {}, Spelling S = GNU_swift_async);
  static SwiftAsyncAttr *Create(ASTContext &Ctx, SwiftAsyncAttr::Kind Kind, ParamIdx CompletionHandlerIndex, SourceRange Range = {}, Spelling S = GNU_swift_async);

  // Constructors
  SwiftAsyncAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftAsyncAttr::Kind Kind
              , ParamIdx CompletionHandlerIndex
             );
  SwiftAsyncAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftAsyncAttr::Kind Kind
             );

  SwiftAsyncAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  SwiftAsyncAttr::Kind getKind() const {
    return kind;
  }

  static bool ConvertStrToKind(StringRef Val, SwiftAsyncAttr::Kind &Out);
  static const char *ConvertKindToStr(SwiftAsyncAttr::Kind Val);
  ParamIdx getCompletionHandlerIndex() const {
    return completionHandlerIndex;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAsync; }
};

class CLANG_ABI SwiftAsyncCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_swiftasynccall = 0,
    CXX11_clang_swiftasynccall = 1,
    C23_clang_swiftasynccall = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftAsyncCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swiftasynccall);
  static SwiftAsyncCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swiftasynccall);

  // Constructors
  SwiftAsyncCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftAsyncCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAsyncCall; }
};

class CLANG_ABI SwiftAsyncContextAttr : public ParameterABIAttr {
public:
  enum Spelling {
    GNU_swift_async_context = 0,
    CXX11_clang_swift_async_context = 1,
    C23_clang_swift_async_context = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftAsyncContextAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncContextAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncContextAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_async_context);
  static SwiftAsyncContextAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_async_context);

  // Constructors
  SwiftAsyncContextAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftAsyncContextAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAsyncContext; }
};

class CLANG_ABI SwiftAsyncErrorAttr : public InheritableAttr {
public:
  enum ConventionKind {
    None,
    NonNullError,
    ZeroArgument,
    NonZeroArgument
  };
private:
  SwiftAsyncErrorAttr::ConventionKind convention;

unsigned handlerParamIdx;

public:
  enum Spelling {
    GNU_swift_async_error = 0,
    CXX11_clang_swift_async_error = 1,
    C23_clang_swift_async_error = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftAsyncErrorAttr *CreateImplicit(ASTContext &Ctx, SwiftAsyncErrorAttr::ConventionKind Convention, unsigned HandlerParamIdx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncErrorAttr *Create(ASTContext &Ctx, SwiftAsyncErrorAttr::ConventionKind Convention, unsigned HandlerParamIdx, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncErrorAttr *CreateImplicit(ASTContext &Ctx, SwiftAsyncErrorAttr::ConventionKind Convention, unsigned HandlerParamIdx, SourceRange Range = {}, Spelling S = GNU_swift_async_error);
  static SwiftAsyncErrorAttr *Create(ASTContext &Ctx, SwiftAsyncErrorAttr::ConventionKind Convention, unsigned HandlerParamIdx, SourceRange Range = {}, Spelling S = GNU_swift_async_error);

  // Constructors
  SwiftAsyncErrorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftAsyncErrorAttr::ConventionKind Convention
              , unsigned HandlerParamIdx
             );
  SwiftAsyncErrorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftAsyncErrorAttr::ConventionKind Convention
             );

  SwiftAsyncErrorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  SwiftAsyncErrorAttr::ConventionKind getConvention() const {
    return convention;
  }

  static bool ConvertStrToConventionKind(StringRef Val, SwiftAsyncErrorAttr::ConventionKind &Out);
  static const char *ConvertConventionKindToStr(SwiftAsyncErrorAttr::ConventionKind Val);
  unsigned getHandlerParamIdx() const {
    return handlerParamIdx;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAsyncError; }
};

class CLANG_ABI SwiftAsyncNameAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static SwiftAsyncNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncNameAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SwiftAsyncNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static SwiftAsyncNameAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  SwiftAsyncNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  SwiftAsyncNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAsyncName; }
};

class CLANG_ABI SwiftAttrAttr : public InheritableAttr {
unsigned attributeLength;
char *attribute;

public:
  // Factory methods
  static SwiftAttrAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Attribute, const AttributeCommonInfo &CommonInfo);
  static SwiftAttrAttr *Create(ASTContext &Ctx, llvm::StringRef Attribute, const AttributeCommonInfo &CommonInfo);
  static SwiftAttrAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Attribute, SourceRange Range = {});
  static SwiftAttrAttr *Create(ASTContext &Ctx, llvm::StringRef Attribute, SourceRange Range = {});

  // Constructors
  SwiftAttrAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Attribute
             );

  SwiftAttrAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAttribute() const {
    return llvm::StringRef(attribute, attributeLength);
  }
  unsigned getAttributeLength() const {
    return attributeLength;
  }
  void setAttribute(ASTContext &C, llvm::StringRef S) {
    attributeLength = S.size();
    this->attribute = new (C, 1) char [attributeLength];
    if (!S.empty())
      std::memcpy(this->attribute, S.data(), attributeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftAttr; }
};

class CLANG_ABI SwiftBridgeAttr : public InheritableAttr {
unsigned swiftTypeLength;
char *swiftType;

public:
  // Factory methods
  static SwiftBridgeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef SwiftType, const AttributeCommonInfo &CommonInfo);
  static SwiftBridgeAttr *Create(ASTContext &Ctx, llvm::StringRef SwiftType, const AttributeCommonInfo &CommonInfo);
  static SwiftBridgeAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef SwiftType, SourceRange Range = {});
  static SwiftBridgeAttr *Create(ASTContext &Ctx, llvm::StringRef SwiftType, SourceRange Range = {});

  // Constructors
  SwiftBridgeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef SwiftType
             );

  SwiftBridgeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getSwiftType() const {
    return llvm::StringRef(swiftType, swiftTypeLength);
  }
  unsigned getSwiftTypeLength() const {
    return swiftTypeLength;
  }
  void setSwiftType(ASTContext &C, llvm::StringRef S) {
    swiftTypeLength = S.size();
    this->swiftType = new (C, 1) char [swiftTypeLength];
    if (!S.empty())
      std::memcpy(this->swiftType, S.data(), swiftTypeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftBridge; }
};

class CLANG_ABI SwiftBridgedTypedefAttr : public InheritableAttr {
public:
  // Factory methods
  static SwiftBridgedTypedefAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftBridgedTypedefAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftBridgedTypedefAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SwiftBridgedTypedefAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SwiftBridgedTypedefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftBridgedTypedefAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftBridgedTypedef; }
};

class CLANG_ABI SwiftCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_swiftcall = 0,
    CXX11_clang_swiftcall = 1,
    C23_clang_swiftcall = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swiftcall);
  static SwiftCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swiftcall);

  // Constructors
  SwiftCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftCall; }
};

class CLANG_ABI SwiftContextAttr : public ParameterABIAttr {
public:
  enum Spelling {
    GNU_swift_context = 0,
    CXX11_clang_swift_context = 1,
    C23_clang_swift_context = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftContextAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftContextAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftContextAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_context);
  static SwiftContextAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_context);

  // Constructors
  SwiftContextAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftContextAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftContext; }
};

class CLANG_ABI SwiftErrorAttr : public InheritableAttr {
public:
  enum ConventionKind {
    None,
    NonNullError,
    NullResult,
    ZeroResult,
    NonZeroResult
  };
private:
  SwiftErrorAttr::ConventionKind convention;

public:
  // Factory methods
  static SwiftErrorAttr *CreateImplicit(ASTContext &Ctx, SwiftErrorAttr::ConventionKind Convention, const AttributeCommonInfo &CommonInfo);
  static SwiftErrorAttr *Create(ASTContext &Ctx, SwiftErrorAttr::ConventionKind Convention, const AttributeCommonInfo &CommonInfo);
  static SwiftErrorAttr *CreateImplicit(ASTContext &Ctx, SwiftErrorAttr::ConventionKind Convention, SourceRange Range = {});
  static SwiftErrorAttr *Create(ASTContext &Ctx, SwiftErrorAttr::ConventionKind Convention, SourceRange Range = {});

  // Constructors
  SwiftErrorAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftErrorAttr::ConventionKind Convention
             );

  SwiftErrorAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  SwiftErrorAttr::ConventionKind getConvention() const {
    return convention;
  }

  static bool ConvertStrToConventionKind(StringRef Val, SwiftErrorAttr::ConventionKind &Out);
  static const char *ConvertConventionKindToStr(SwiftErrorAttr::ConventionKind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftError; }
};

class CLANG_ABI SwiftErrorResultAttr : public ParameterABIAttr {
public:
  enum Spelling {
    GNU_swift_error_result = 0,
    CXX11_clang_swift_error_result = 1,
    C23_clang_swift_error_result = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftErrorResultAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftErrorResultAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftErrorResultAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_error_result);
  static SwiftErrorResultAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_error_result);

  // Constructors
  SwiftErrorResultAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftErrorResultAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftErrorResult; }
};

class CLANG_ABI SwiftImportAsNonGenericAttr : public InheritableAttr {
public:
  // Factory methods
  static SwiftImportAsNonGenericAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftImportAsNonGenericAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftImportAsNonGenericAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SwiftImportAsNonGenericAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SwiftImportAsNonGenericAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftImportAsNonGenericAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftImportAsNonGeneric; }
};

class CLANG_ABI SwiftImportPropertyAsAccessorsAttr : public InheritableAttr {
public:
  // Factory methods
  static SwiftImportPropertyAsAccessorsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftImportPropertyAsAccessorsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftImportPropertyAsAccessorsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SwiftImportPropertyAsAccessorsAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SwiftImportPropertyAsAccessorsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftImportPropertyAsAccessorsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftImportPropertyAsAccessors; }
};

class CLANG_ABI SwiftIndirectResultAttr : public ParameterABIAttr {
public:
  enum Spelling {
    GNU_swift_indirect_result = 0,
    CXX11_clang_swift_indirect_result = 1,
    C23_clang_swift_indirect_result = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftIndirectResultAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftIndirectResultAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftIndirectResultAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_indirect_result);
  static SwiftIndirectResultAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_swift_indirect_result);

  // Constructors
  SwiftIndirectResultAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftIndirectResultAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftIndirectResult; }
};

class CLANG_ABI SwiftNameAttr : public InheritableAttr {
unsigned nameLength;
char *name;

public:
  // Factory methods
  static SwiftNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SwiftNameAttr *Create(ASTContext &Ctx, llvm::StringRef Name, const AttributeCommonInfo &CommonInfo);
  static SwiftNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});
  static SwiftNameAttr *Create(ASTContext &Ctx, llvm::StringRef Name, SourceRange Range = {});

  // Constructors
  SwiftNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Name
             );

  SwiftNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getName() const {
    return llvm::StringRef(name, nameLength);
  }
  unsigned getNameLength() const {
    return nameLength;
  }
  void setName(ASTContext &C, llvm::StringRef S) {
    nameLength = S.size();
    this->name = new (C, 1) char [nameLength];
    if (!S.empty())
      std::memcpy(this->name, S.data(), nameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftName; }
};

class CLANG_ABI SwiftNewTypeAttr : public InheritableAttr {
public:
  enum NewtypeKind {
    NK_Struct,
    NK_Enum
  };
private:
  SwiftNewTypeAttr::NewtypeKind newtypeKind;

public:
  enum Spelling {
    GNU_swift_newtype = 0,
    GNU_swift_wrapper = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SwiftNewTypeAttr *CreateImplicit(ASTContext &Ctx, SwiftNewTypeAttr::NewtypeKind NewtypeKind, const AttributeCommonInfo &CommonInfo);
  static SwiftNewTypeAttr *Create(ASTContext &Ctx, SwiftNewTypeAttr::NewtypeKind NewtypeKind, const AttributeCommonInfo &CommonInfo);
  static SwiftNewTypeAttr *CreateImplicit(ASTContext &Ctx, SwiftNewTypeAttr::NewtypeKind NewtypeKind, SourceRange Range = {}, Spelling S = GNU_swift_newtype);
  static SwiftNewTypeAttr *Create(ASTContext &Ctx, SwiftNewTypeAttr::NewtypeKind NewtypeKind, SourceRange Range = {}, Spelling S = GNU_swift_newtype);

  // Constructors
  SwiftNewTypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , SwiftNewTypeAttr::NewtypeKind NewtypeKind
             );

  SwiftNewTypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  SwiftNewTypeAttr::NewtypeKind getNewtypeKind() const {
    return newtypeKind;
  }

  static bool ConvertStrToNewtypeKind(StringRef Val, SwiftNewTypeAttr::NewtypeKind &Out);
  static const char *ConvertNewtypeKindToStr(SwiftNewTypeAttr::NewtypeKind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftNewType; }
};

class CLANG_ABI SwiftObjCMembersAttr : public Attr {
public:
  // Factory methods
  static SwiftObjCMembersAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftObjCMembersAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftObjCMembersAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SwiftObjCMembersAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SwiftObjCMembersAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftObjCMembersAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftObjCMembers; }
};

class CLANG_ABI SwiftPrivateAttr : public InheritableAttr {
public:
  // Factory methods
  static SwiftPrivateAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftPrivateAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SwiftPrivateAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static SwiftPrivateAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  SwiftPrivateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SwiftPrivateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftPrivate; }
};

class CLANG_ABI SwiftVersionedAdditionAttr : public Attr {
VersionTuple version;


Attr * additionalAttr;

bool isReplacedByActive;

public:
  // Factory methods
  static SwiftVersionedAdditionAttr *CreateImplicit(ASTContext &Ctx, VersionTuple Version, Attr * AdditionalAttr, bool IsReplacedByActive, const AttributeCommonInfo &CommonInfo);
  static SwiftVersionedAdditionAttr *Create(ASTContext &Ctx, VersionTuple Version, Attr * AdditionalAttr, bool IsReplacedByActive, const AttributeCommonInfo &CommonInfo);
  static SwiftVersionedAdditionAttr *CreateImplicit(ASTContext &Ctx, VersionTuple Version, Attr * AdditionalAttr, bool IsReplacedByActive, SourceRange Range = {});
  static SwiftVersionedAdditionAttr *Create(ASTContext &Ctx, VersionTuple Version, Attr * AdditionalAttr, bool IsReplacedByActive, SourceRange Range = {});

  // Constructors
  SwiftVersionedAdditionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , VersionTuple Version
              , Attr * AdditionalAttr
              , bool IsReplacedByActive
             );

  SwiftVersionedAdditionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  VersionTuple getVersion() const {
    return version;
  }
  void setVersion(ASTContext &C, VersionTuple V) {
    version = V;
  }

  Attr * getAdditionalAttr() const {
    return additionalAttr;
  }

  bool getIsReplacedByActive() const {
    return isReplacedByActive;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftVersionedAddition; }
};

class CLANG_ABI SwiftVersionedRemovalAttr : public Attr {
VersionTuple version;


unsigned rawKind;

bool isReplacedByActive;

public:
  // Factory methods
  static SwiftVersionedRemovalAttr *CreateImplicit(ASTContext &Ctx, VersionTuple Version, unsigned RawKind, bool IsReplacedByActive, const AttributeCommonInfo &CommonInfo);
  static SwiftVersionedRemovalAttr *Create(ASTContext &Ctx, VersionTuple Version, unsigned RawKind, bool IsReplacedByActive, const AttributeCommonInfo &CommonInfo);
  static SwiftVersionedRemovalAttr *CreateImplicit(ASTContext &Ctx, VersionTuple Version, unsigned RawKind, bool IsReplacedByActive, SourceRange Range = {});
  static SwiftVersionedRemovalAttr *Create(ASTContext &Ctx, VersionTuple Version, unsigned RawKind, bool IsReplacedByActive, SourceRange Range = {});

  // Constructors
  SwiftVersionedRemovalAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , VersionTuple Version
              , unsigned RawKind
              , bool IsReplacedByActive
             );

  SwiftVersionedRemovalAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  VersionTuple getVersion() const {
    return version;
  }
  void setVersion(ASTContext &C, VersionTuple V) {
    version = V;
  }

  unsigned getRawKind() const {
    return rawKind;
  }

  bool getIsReplacedByActive() const {
    return isReplacedByActive;
  }


    attr::Kind getAttrKindToRemove() const {
      return static_cast<attr::Kind>(getRawKind());
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::SwiftVersionedRemoval; }
};

class CLANG_ABI SysVABIAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_sysv_abi = 0,
    CXX11_gnu_sysv_abi = 1,
    C23_gnu_sysv_abi = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static SysVABIAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SysVABIAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static SysVABIAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sysv_abi);
  static SysVABIAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_sysv_abi);

  // Constructors
  SysVABIAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  SysVABIAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::SysVABI; }
};

class CLANG_ABI TLSModelAttr : public InheritableAttr {
unsigned modelLength;
char *model;

public:
  enum Spelling {
    GNU_tls_model = 0,
    CXX11_gnu_tls_model = 1,
    C23_gnu_tls_model = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TLSModelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Model, const AttributeCommonInfo &CommonInfo);
  static TLSModelAttr *Create(ASTContext &Ctx, llvm::StringRef Model, const AttributeCommonInfo &CommonInfo);
  static TLSModelAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Model, SourceRange Range = {}, Spelling S = GNU_tls_model);
  static TLSModelAttr *Create(ASTContext &Ctx, llvm::StringRef Model, SourceRange Range = {}, Spelling S = GNU_tls_model);

  // Constructors
  TLSModelAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Model
             );

  TLSModelAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getModel() const {
    return llvm::StringRef(model, modelLength);
  }
  unsigned getModelLength() const {
    return modelLength;
  }
  void setModel(ASTContext &C, llvm::StringRef S) {
    modelLength = S.size();
    this->model = new (C, 1) char [modelLength];
    if (!S.empty())
      std::memcpy(this->model, S.data(), modelLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::TLSModel; }
};

class CLANG_ABI TargetAttr : public InheritableAttr {
unsigned featuresStrLength;
char *featuresStr;

public:
  enum Spelling {
    GNU_target = 0,
    CXX11_gnu_target = 1,
    C23_gnu_target = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TargetAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef FeaturesStr, const AttributeCommonInfo &CommonInfo);
  static TargetAttr *Create(ASTContext &Ctx, llvm::StringRef FeaturesStr, const AttributeCommonInfo &CommonInfo);
  static TargetAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef FeaturesStr, SourceRange Range = {}, Spelling S = GNU_target);
  static TargetAttr *Create(ASTContext &Ctx, llvm::StringRef FeaturesStr, SourceRange Range = {}, Spelling S = GNU_target);

  // Constructors
  TargetAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef FeaturesStr
             );

  TargetAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getFeaturesStr() const {
    return llvm::StringRef(featuresStr, featuresStrLength);
  }
  unsigned getFeaturesStrLength() const {
    return featuresStrLength;
  }
  void setFeaturesStr(ASTContext &C, llvm::StringRef S) {
    featuresStrLength = S.size();
    this->featuresStr = new (C, 1) char [featuresStrLength];
    if (!S.empty())
      std::memcpy(this->featuresStr, S.data(), featuresStrLength);
  }


    std::optional<StringRef> getX86Architecture() const {
      StringRef Features = getFeaturesStr();
      SmallVector<StringRef, 4> AttrFeatures;
      Features.split(AttrFeatures, ',');
      for (StringRef Feature : AttrFeatures) {
        Feature = Feature.trim();
        if (Feature.starts_with("arch="))
          return Feature.drop_front(sizeof("arch=") - 1);
      }
      return std::nullopt;
    }

    // Gets the list of features as simple string-refs with no +/- or 'no-'.
    // Only adds the items to 'Out' that are additions.
    void getX86AddedFeatures(llvm::SmallVectorImpl<StringRef> &Out) const {
      if (isDefaultVersion())
        return;
      StringRef Features = getFeaturesStr();
      SmallVector<StringRef, 4> AttrFeatures;
      Features.split(AttrFeatures, ',');
      for (auto &Feature : AttrFeatures) {
        Feature = Feature.trim();
        if (!Feature.starts_with("no-") && !Feature.starts_with("arch=") &&
            !Feature.starts_with("fpmath=") && !Feature.starts_with("tune="))
          Out.push_back(Feature);
      }
    }

    bool isDefaultVersion() const { return getFeaturesStr() == "default"; }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::Target; }
};

class CLANG_ABI TargetClonesAttr : public InheritableAttr {
  unsigned featuresStrs_Size;
  StringRef *featuresStrs_;

public:
  enum Spelling {
    GNU_target_clones = 0,
    CXX11_gnu_target_clones = 1,
    C23_gnu_target_clones = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TargetClonesAttr *CreateImplicit(ASTContext &Ctx, StringRef *FeaturesStrs, unsigned FeaturesStrsSize, const AttributeCommonInfo &CommonInfo);
  static TargetClonesAttr *Create(ASTContext &Ctx, StringRef *FeaturesStrs, unsigned FeaturesStrsSize, const AttributeCommonInfo &CommonInfo);
  static TargetClonesAttr *CreateImplicit(ASTContext &Ctx, StringRef *FeaturesStrs, unsigned FeaturesStrsSize, SourceRange Range = {}, Spelling S = GNU_target_clones);
  static TargetClonesAttr *Create(ASTContext &Ctx, StringRef *FeaturesStrs, unsigned FeaturesStrsSize, SourceRange Range = {}, Spelling S = GNU_target_clones);

  // Constructors
  TargetClonesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , StringRef *FeaturesStrs, unsigned FeaturesStrsSize
             );
  TargetClonesAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TargetClonesAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  typedef StringRef* featuresStrs_iterator;
  featuresStrs_iterator featuresStrs_begin() const { return featuresStrs_; }
  featuresStrs_iterator featuresStrs_end() const { return featuresStrs_ + featuresStrs_Size; }
  unsigned featuresStrs_size() const { return featuresStrs_Size; }
  llvm::iterator_range<featuresStrs_iterator> featuresStrs() const { return llvm::make_range(featuresStrs_begin(), featuresStrs_end()); }



    StringRef getFeatureStr(unsigned Index) const {
      return *(featuresStrs_begin() + Index);
    }

    bool isDefaultVersion(unsigned Index) const {
      return getFeatureStr(Index) == "default";
    }

    void getFeatures(llvm::SmallVectorImpl<StringRef> &Out,
                     unsigned Index, char Delim = '+') const {
      if (isDefaultVersion(Index))
        return;
      StringRef Features = getFeatureStr(Index);
      SmallVector<StringRef, 4> AttrFeatures;
      Features.split(AttrFeatures, Delim);
      for (StringRef Feature : AttrFeatures) {
        Feature = Feature.trim();
        Out.push_back(Feature);
      }
    }

    std::optional<StringRef> getX86Architecture(unsigned Index) const {
      StringRef Feature = getFeatureStr(Index);
      if (Feature.starts_with("arch="))
        return Feature.drop_front(sizeof("arch=") - 1);
      return std::nullopt;
    }

    void getX86Feature(llvm::SmallVectorImpl<StringRef> &Out,
                       unsigned Index) const {
      if (isDefaultVersion(Index))
        return;
      if (getX86Architecture(Index))
        return;
      Out.push_back(getFeatureStr(Index));
    }

    // Given an index into the 'featuresStrs' sequence, compute a unique
    // ID to be used with function name mangling for the associated variant.
    // This mapping is necessary due to a requirement that the mangling ID
    // used for the "default" variant be the largest mangling ID in the
    // variant set. Duplicate variants present in 'featuresStrs' are also
    // assigned their own unique ID (the mapping is bijective).
    unsigned getMangledIndex(unsigned Index) const {
      if (getFeatureStr(Index) == "default")
        return std::count_if(featuresStrs_begin(), featuresStrs_end(),
                              [](StringRef S) { return S != "default"; });

      return std::count_if(featuresStrs_begin(), featuresStrs_begin() + Index,
                           [](StringRef S) { return S != "default"; });
    }

    // Given an index into the 'featuresStrs' sequence, determine if the
    // index corresponds to the first instance of the named variant. This
    // is used to skip over duplicate variant instances when iterating over
    // 'featuresStrs'.
    bool isFirstOfVersion(unsigned Index) const {
      StringRef FeatureStr(getFeatureStr(Index));
      return 0 == std::count_if(
                      featuresStrs_begin(), featuresStrs_begin() + Index,
                      [FeatureStr](StringRef S) { return S == FeatureStr; });

    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::TargetClones; }
};

class CLANG_ABI TargetVersionAttr : public InheritableAttr {
unsigned namesStrLength;
char *namesStr;

public:
  enum Spelling {
    GNU_target_version = 0,
    CXX11_gnu_target_version = 1,
    C23_gnu_target_version = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TargetVersionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef NamesStr, const AttributeCommonInfo &CommonInfo);
  static TargetVersionAttr *Create(ASTContext &Ctx, llvm::StringRef NamesStr, const AttributeCommonInfo &CommonInfo);
  static TargetVersionAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef NamesStr, SourceRange Range = {}, Spelling S = GNU_target_version);
  static TargetVersionAttr *Create(ASTContext &Ctx, llvm::StringRef NamesStr, SourceRange Range = {}, Spelling S = GNU_target_version);

  // Constructors
  TargetVersionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef NamesStr
             );

  TargetVersionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getNamesStr() const {
    return llvm::StringRef(namesStr, namesStrLength);
  }
  unsigned getNamesStrLength() const {
    return namesStrLength;
  }
  void setNamesStr(ASTContext &C, llvm::StringRef S) {
    namesStrLength = S.size();
    this->namesStr = new (C, 1) char [namesStrLength];
    if (!S.empty())
      std::memcpy(this->namesStr, S.data(), namesStrLength);
  }


    StringRef getName() const { return getNamesStr().trim(); }

    bool isDefaultVersion() const { return getName() == "default"; }

    void getFeatures(llvm::SmallVectorImpl<StringRef> &Out,
                     char Delim = '+') const {
      if (isDefaultVersion())
        return;
      StringRef Features = getName();
      SmallVector<StringRef, 4> AttrFeatures;
      Features.split(AttrFeatures, Delim);
      for (StringRef Feature : AttrFeatures) {
        Feature = Feature.trim();
        Out.push_back(Feature);
      }
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::TargetVersion; }
};

class CLANG_ABI TestTypestateAttr : public InheritableAttr {
public:
  enum ConsumedState {
    Consumed,
    Unconsumed
  };
private:
  TestTypestateAttr::ConsumedState testState;

public:
  enum Spelling {
    GNU_test_typestate = 0,
    CXX11_clang_test_typestate = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TestTypestateAttr *CreateImplicit(ASTContext &Ctx, TestTypestateAttr::ConsumedState TestState, const AttributeCommonInfo &CommonInfo);
  static TestTypestateAttr *Create(ASTContext &Ctx, TestTypestateAttr::ConsumedState TestState, const AttributeCommonInfo &CommonInfo);
  static TestTypestateAttr *CreateImplicit(ASTContext &Ctx, TestTypestateAttr::ConsumedState TestState, SourceRange Range = {}, Spelling S = GNU_test_typestate);
  static TestTypestateAttr *Create(ASTContext &Ctx, TestTypestateAttr::ConsumedState TestState, SourceRange Range = {}, Spelling S = GNU_test_typestate);

  // Constructors
  TestTypestateAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TestTypestateAttr::ConsumedState TestState
             );

  TestTypestateAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  TestTypestateAttr::ConsumedState getTestState() const {
    return testState;
  }

  static bool ConvertStrToConsumedState(StringRef Val, TestTypestateAttr::ConsumedState &Out);
  static const char *ConvertConsumedStateToStr(TestTypestateAttr::ConsumedState Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::TestTypestate; }
};

class CLANG_ABI ThisCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_thiscall = 0,
    CXX11_gnu_thiscall = 1,
    C23_gnu_thiscall = 2,
    Keyword_thiscall = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ThisCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ThisCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ThisCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_thiscall);
  static ThisCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_thiscall);

  // Constructors
  ThisCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ThisCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::ThisCall; }
};

class CLANG_ABI ThreadAttr : public Attr {
public:
  // Factory methods
  static ThreadAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ThreadAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static ThreadAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static ThreadAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  ThreadAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  ThreadAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Thread; }
};

class CLANG_ABI TransparentUnionAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_transparent_union = 0,
    CXX11_gnu_transparent_union = 1,
    C23_gnu_transparent_union = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TransparentUnionAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TransparentUnionAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TransparentUnionAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_transparent_union);
  static TransparentUnionAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_transparent_union);

  // Constructors
  TransparentUnionAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TransparentUnionAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TransparentUnion; }
};

class CLANG_ABI TrivialABIAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_trivial_abi = 0,
    CXX11_clang_trivial_abi = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TrivialABIAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TrivialABIAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TrivialABIAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_trivial_abi);
  static TrivialABIAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_trivial_abi);

  // Constructors
  TrivialABIAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TrivialABIAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TrivialABI; }
};

class CLANG_ABI TryAcquireCapabilityAttr : public InheritableAttr {
Expr * successValue;

  unsigned args_Size;
  Expr * *args_;

public:
  enum Spelling {
    GNU_try_acquire_capability = 0,
    CXX11_clang_try_acquire_capability = 1,
    GNU_try_acquire_shared_capability = 2,
    CXX11_clang_try_acquire_shared_capability = 3,
    GNU_exclusive_trylock_function = 4,
    GNU_shared_trylock_function = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TryAcquireCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * SuccessValue, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static TryAcquireCapabilityAttr *Create(ASTContext &Ctx, Expr * SuccessValue, Expr * *Args, unsigned ArgsSize, const AttributeCommonInfo &CommonInfo);
  static TryAcquireCapabilityAttr *CreateImplicit(ASTContext &Ctx, Expr * SuccessValue, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_try_acquire_capability);
  static TryAcquireCapabilityAttr *Create(ASTContext &Ctx, Expr * SuccessValue, Expr * *Args, unsigned ArgsSize, SourceRange Range = {}, Spelling S = GNU_try_acquire_capability);

  // Constructors
  TryAcquireCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * SuccessValue
              , Expr * *Args, unsigned ArgsSize
             );
  TryAcquireCapabilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * SuccessValue
             );

  TryAcquireCapabilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool isShared() const { return getAttributeSpellingListIndex() == 2 ||
    getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 5; }
  Expr * getSuccessValue() const {
    return successValue;
  }

  typedef Expr ** args_iterator;
  args_iterator args_begin() const { return args_; }
  args_iterator args_end() const { return args_ + args_Size; }
  unsigned args_size() const { return args_Size; }
  llvm::iterator_range<args_iterator> args() const { return llvm::make_range(args_begin(), args_end()); }




  static bool classof(const Attr *A) { return A->getKind() == attr::TryAcquireCapability; }
};

class CLANG_ABI TypeNonNullAttr : public TypeAttr {
public:
  // Factory methods
  static TypeNonNullAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNonNullAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNonNullAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static TypeNonNullAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  TypeNonNullAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TypeNonNullAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TypeNonNull; }
};

class CLANG_ABI TypeNullUnspecifiedAttr : public TypeAttr {
public:
  // Factory methods
  static TypeNullUnspecifiedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullUnspecifiedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullUnspecifiedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static TypeNullUnspecifiedAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  TypeNullUnspecifiedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TypeNullUnspecifiedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TypeNullUnspecified; }
};

class CLANG_ABI TypeNullableAttr : public InheritableAttr {
public:
  // Factory methods
  static TypeNullableAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullableAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullableAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static TypeNullableAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  TypeNullableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TypeNullableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TypeNullable; }
};

class CLANG_ABI TypeNullableResultAttr : public TypeAttr {
public:
  // Factory methods
  static TypeNullableResultAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullableResultAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static TypeNullableResultAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static TypeNullableResultAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  TypeNullableResultAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  TypeNullableResultAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::TypeNullableResult; }
};

class CLANG_ABI TypeTagForDatatypeAttr : public InheritableAttr {
IdentifierInfo * argumentKind;

TypeSourceInfo * matchingCType;

bool layoutCompatible;

bool mustBeNull;

public:
  enum Spelling {
    GNU_type_tag_for_datatype = 0,
    CXX11_clang_type_tag_for_datatype = 1,
    C23_clang_type_tag_for_datatype = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TypeTagForDatatypeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, TypeSourceInfo * MatchingCType, bool LayoutCompatible, bool MustBeNull, const AttributeCommonInfo &CommonInfo);
  static TypeTagForDatatypeAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, TypeSourceInfo * MatchingCType, bool LayoutCompatible, bool MustBeNull, const AttributeCommonInfo &CommonInfo);
  static TypeTagForDatatypeAttr *CreateImplicit(ASTContext &Ctx, IdentifierInfo * ArgumentKind, TypeSourceInfo * MatchingCType, bool LayoutCompatible, bool MustBeNull, SourceRange Range = {}, Spelling S = GNU_type_tag_for_datatype);
  static TypeTagForDatatypeAttr *Create(ASTContext &Ctx, IdentifierInfo * ArgumentKind, TypeSourceInfo * MatchingCType, bool LayoutCompatible, bool MustBeNull, SourceRange Range = {}, Spelling S = GNU_type_tag_for_datatype);

  // Constructors
  TypeTagForDatatypeAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , IdentifierInfo * ArgumentKind
              , TypeSourceInfo * MatchingCType
              , bool LayoutCompatible
              , bool MustBeNull
             );

  TypeTagForDatatypeAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  IdentifierInfo * getArgumentKind() const {
    return argumentKind;
  }

  QualType getMatchingCType() const {
    return matchingCType->getType();
  }  TypeSourceInfo * getMatchingCTypeLoc() const {
    return matchingCType;
  }

  bool getLayoutCompatible() const {
    return layoutCompatible;
  }

  bool getMustBeNull() const {
    return mustBeNull;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::TypeTagForDatatype; }
};

class CLANG_ABI TypeVisibilityAttr : public InheritableAttr {
public:
  enum VisibilityType {
    Default,
    Hidden,
    Protected
  };
private:
  TypeVisibilityAttr::VisibilityType visibility;

public:
  enum Spelling {
    GNU_type_visibility = 0,
    CXX11_clang_type_visibility = 1,
    C23_clang_type_visibility = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static TypeVisibilityAttr *CreateImplicit(ASTContext &Ctx, TypeVisibilityAttr::VisibilityType Visibility, const AttributeCommonInfo &CommonInfo);
  static TypeVisibilityAttr *Create(ASTContext &Ctx, TypeVisibilityAttr::VisibilityType Visibility, const AttributeCommonInfo &CommonInfo);
  static TypeVisibilityAttr *CreateImplicit(ASTContext &Ctx, TypeVisibilityAttr::VisibilityType Visibility, SourceRange Range = {}, Spelling S = GNU_type_visibility);
  static TypeVisibilityAttr *Create(ASTContext &Ctx, TypeVisibilityAttr::VisibilityType Visibility, SourceRange Range = {}, Spelling S = GNU_type_visibility);

  // Constructors
  TypeVisibilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeVisibilityAttr::VisibilityType Visibility
             );

  TypeVisibilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  TypeVisibilityAttr::VisibilityType getVisibility() const {
    return visibility;
  }

  static bool ConvertStrToVisibilityType(StringRef Val, TypeVisibilityAttr::VisibilityType &Out);
  static const char *ConvertVisibilityTypeToStr(TypeVisibilityAttr::VisibilityType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::TypeVisibility; }
};

class CLANG_ABI UPtrAttr : public TypeAttr {
public:
  // Factory methods
  static UPtrAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UPtrAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UPtrAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static UPtrAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  UPtrAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UPtrAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::UPtr; }
};

class CLANG_ABI UnavailableAttr : public InheritableAttr {
unsigned messageLength;
char *message;

public:
  enum ImplicitReason {
    IR_None,
    IR_ARCForbiddenType,
    IR_ForbiddenWeak,
    IR_ARCForbiddenConversion,
    IR_ARCInitReturnsUnrelated,
    IR_ARCFieldWithOwnership
  };
private:
  UnavailableAttr::ImplicitReason implicitReason;

public:
  enum Spelling {
    GNU_unavailable = 0,
    CXX11_clang_unavailable = 1,
    C23_clang_unavailable = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UnavailableAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, UnavailableAttr::ImplicitReason ImplicitReason, const AttributeCommonInfo &CommonInfo);
  static UnavailableAttr *Create(ASTContext &Ctx, llvm::StringRef Message, UnavailableAttr::ImplicitReason ImplicitReason, const AttributeCommonInfo &CommonInfo);
  static UnavailableAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, UnavailableAttr::ImplicitReason ImplicitReason, SourceRange Range = {}, Spelling S = GNU_unavailable);
  static UnavailableAttr *Create(ASTContext &Ctx, llvm::StringRef Message, UnavailableAttr::ImplicitReason ImplicitReason, SourceRange Range = {}, Spelling S = GNU_unavailable);
  static UnavailableAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static UnavailableAttr *Create(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static UnavailableAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = GNU_unavailable);
  static UnavailableAttr *Create(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = GNU_unavailable);

  // Constructors
  UnavailableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Message
              , UnavailableAttr::ImplicitReason ImplicitReason
             );
  UnavailableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Message
             );
  UnavailableAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UnavailableAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }

  UnavailableAttr::ImplicitReason getImplicitReason() const {
    return implicitReason;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Unavailable; }
};

class CLANG_ABI UninitializedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_uninitialized = 0,
    CXX11_clang_uninitialized = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UninitializedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UninitializedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UninitializedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_uninitialized);
  static UninitializedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_uninitialized);

  // Constructors
  UninitializedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UninitializedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Uninitialized; }
};

class CLANG_ABI UnlikelyAttr : public StmtAttr {
public:
  enum Spelling {
    CXX11_unlikely = 0,
    C23_clang_unlikely = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UnlikelyAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnlikelyAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnlikelyAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_unlikely);
  static UnlikelyAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_unlikely);

  // Constructors
  UnlikelyAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UnlikelyAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Unlikely; }
};

class CLANG_ABI UnsafeBufferUsageAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_unsafe_buffer_usage = 0,
    CXX11_clang_unsafe_buffer_usage = 1,
    C23_clang_unsafe_buffer_usage = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UnsafeBufferUsageAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnsafeBufferUsageAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnsafeBufferUsageAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_unsafe_buffer_usage);
  static UnsafeBufferUsageAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_unsafe_buffer_usage);

  // Constructors
  UnsafeBufferUsageAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UnsafeBufferUsageAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::UnsafeBufferUsage; }
};

class CLANG_ABI UnusedAttr : public InheritableAttr {
public:
  enum Spelling {
    CXX11_maybe_unused = 0,
    GNU_unused = 1,
    CXX11_gnu_unused = 2,
    C23_gnu_unused = 3,
    C23_maybe_unused = 4,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UnusedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnusedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UnusedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_maybe_unused);
  static UnusedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = CXX11_maybe_unused);

  // Constructors
  UnusedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UnusedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Unused; }
};

class CLANG_ABI UseHandleAttr : public InheritableParamAttr {
unsigned handleTypeLength;
char *handleType;

public:
  enum Spelling {
    GNU_use_handle = 0,
    CXX11_clang_use_handle = 1,
    C23_clang_use_handle = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UseHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static UseHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, const AttributeCommonInfo &CommonInfo);
  static UseHandleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_use_handle);
  static UseHandleAttr *Create(ASTContext &Ctx, llvm::StringRef HandleType, SourceRange Range = {}, Spelling S = GNU_use_handle);

  // Constructors
  UseHandleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef HandleType
             );

  UseHandleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getHandleType() const {
    return llvm::StringRef(handleType, handleTypeLength);
  }
  unsigned getHandleTypeLength() const {
    return handleTypeLength;
  }
  void setHandleType(ASTContext &C, llvm::StringRef S) {
    handleTypeLength = S.size();
    this->handleType = new (C, 1) char [handleTypeLength];
    if (!S.empty())
      std::memcpy(this->handleType, S.data(), handleTypeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::UseHandle; }
};

class CLANG_ABI UsedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_used = 0,
    CXX11_gnu_used = 1,
    C23_gnu_used = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UsedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UsedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UsedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_used);
  static UsedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_used);

  // Constructors
  UsedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UsedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Used; }
};

class CLANG_ABI UsingIfExistsAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_using_if_exists = 0,
    CXX11_clang_using_if_exists = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UsingIfExistsAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UsingIfExistsAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static UsingIfExistsAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_using_if_exists);
  static UsingIfExistsAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_using_if_exists);

  // Constructors
  UsingIfExistsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  UsingIfExistsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::UsingIfExists; }
};

class CLANG_ABI UuidAttr : public InheritableAttr {
unsigned guidLength;
char *guid;

MSGuidDecl * guidDecl;

public:
  enum Spelling {
    Declspec_uuid = 0,
    Microsoft_uuid = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static UuidAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Guid, MSGuidDecl * GuidDecl, const AttributeCommonInfo &CommonInfo);
  static UuidAttr *Create(ASTContext &Ctx, llvm::StringRef Guid, MSGuidDecl * GuidDecl, const AttributeCommonInfo &CommonInfo);
  static UuidAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Guid, MSGuidDecl * GuidDecl, SourceRange Range = {}, Spelling S = Declspec_uuid);
  static UuidAttr *Create(ASTContext &Ctx, llvm::StringRef Guid, MSGuidDecl * GuidDecl, SourceRange Range = {}, Spelling S = Declspec_uuid);
  static UuidAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Guid, const AttributeCommonInfo &CommonInfo);
  static UuidAttr *Create(ASTContext &Ctx, llvm::StringRef Guid, const AttributeCommonInfo &CommonInfo);
  static UuidAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Guid, SourceRange Range = {}, Spelling S = Declspec_uuid);
  static UuidAttr *Create(ASTContext &Ctx, llvm::StringRef Guid, SourceRange Range = {}, Spelling S = Declspec_uuid);

  // Constructors
  UuidAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Guid
              , MSGuidDecl * GuidDecl
             );
  UuidAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Guid
             );

  UuidAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getGuid() const {
    return llvm::StringRef(guid, guidLength);
  }
  unsigned getGuidLength() const {
    return guidLength;
  }
  void setGuid(ASTContext &C, llvm::StringRef S) {
    guidLength = S.size();
    this->guid = new (C, 1) char [guidLength];
    if (!S.empty())
      std::memcpy(this->guid, S.data(), guidLength);
  }

  MSGuidDecl * getGuidDecl() const {
    return guidDecl;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::Uuid; }
};

class CLANG_ABI VTablePointerAuthenticationAttr : public InheritableAttr {
public:
  enum VPtrAuthKeyType {
    DefaultKey,
    NoKey,
    ProcessDependent,
    ProcessIndependent
  };
private:
  VTablePointerAuthenticationAttr::VPtrAuthKeyType key;

public:
  enum AddressDiscriminationMode {
    DefaultAddressDiscrimination,
    NoAddressDiscrimination,
    AddressDiscrimination
  };
private:
  VTablePointerAuthenticationAttr::AddressDiscriminationMode addressDiscrimination;

public:
  enum ExtraDiscrimination {
    DefaultExtraDiscrimination,
    NoExtraDiscrimination,
    TypeDiscrimination,
    CustomDiscrimination
  };
private:
  VTablePointerAuthenticationAttr::ExtraDiscrimination extraDiscrimination;

int customDiscriminationValue;

public:
  enum Spelling {
    GNU_ptrauth_vtable_pointer = 0,
    CXX11_clang_ptrauth_vtable_pointer = 1,
    C23_clang_ptrauth_vtable_pointer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static VTablePointerAuthenticationAttr *CreateImplicit(ASTContext &Ctx, VTablePointerAuthenticationAttr::VPtrAuthKeyType Key, VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination, VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination, int CustomDiscriminationValue, const AttributeCommonInfo &CommonInfo);
  static VTablePointerAuthenticationAttr *Create(ASTContext &Ctx, VTablePointerAuthenticationAttr::VPtrAuthKeyType Key, VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination, VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination, int CustomDiscriminationValue, const AttributeCommonInfo &CommonInfo);
  static VTablePointerAuthenticationAttr *CreateImplicit(ASTContext &Ctx, VTablePointerAuthenticationAttr::VPtrAuthKeyType Key, VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination, VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination, int CustomDiscriminationValue, SourceRange Range = {}, Spelling S = GNU_ptrauth_vtable_pointer);
  static VTablePointerAuthenticationAttr *Create(ASTContext &Ctx, VTablePointerAuthenticationAttr::VPtrAuthKeyType Key, VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination, VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination, int CustomDiscriminationValue, SourceRange Range = {}, Spelling S = GNU_ptrauth_vtable_pointer);

  // Constructors
  VTablePointerAuthenticationAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , VTablePointerAuthenticationAttr::VPtrAuthKeyType Key
              , VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination
              , VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination
              , int CustomDiscriminationValue
             );
  VTablePointerAuthenticationAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , VTablePointerAuthenticationAttr::VPtrAuthKeyType Key
              , VTablePointerAuthenticationAttr::AddressDiscriminationMode AddressDiscrimination
              , VTablePointerAuthenticationAttr::ExtraDiscrimination ExtraDiscrimination
             );

  VTablePointerAuthenticationAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  VTablePointerAuthenticationAttr::VPtrAuthKeyType getKey() const {
    return key;
  }

  static bool ConvertStrToVPtrAuthKeyType(StringRef Val, VTablePointerAuthenticationAttr::VPtrAuthKeyType &Out);
  static const char *ConvertVPtrAuthKeyTypeToStr(VTablePointerAuthenticationAttr::VPtrAuthKeyType Val);
  VTablePointerAuthenticationAttr::AddressDiscriminationMode getAddressDiscrimination() const {
    return addressDiscrimination;
  }

  static bool ConvertStrToAddressDiscriminationMode(StringRef Val, VTablePointerAuthenticationAttr::AddressDiscriminationMode &Out);
  static const char *ConvertAddressDiscriminationModeToStr(VTablePointerAuthenticationAttr::AddressDiscriminationMode Val);
  VTablePointerAuthenticationAttr::ExtraDiscrimination getExtraDiscrimination() const {
    return extraDiscrimination;
  }

  static bool ConvertStrToExtraDiscrimination(StringRef Val, VTablePointerAuthenticationAttr::ExtraDiscrimination &Out);
  static const char *ConvertExtraDiscriminationToStr(VTablePointerAuthenticationAttr::ExtraDiscrimination Val);
  int getCustomDiscriminationValue() const {
    return customDiscriminationValue;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::VTablePointerAuthentication; }
};

class CLANG_ABI VecReturnAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_vecreturn = 0,
    CXX11_clang_vecreturn = 1,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static VecReturnAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static VecReturnAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static VecReturnAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_vecreturn);
  static VecReturnAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_vecreturn);

  // Constructors
  VecReturnAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  VecReturnAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::VecReturn; }
};

class CLANG_ABI VecTypeHintAttr : public InheritableAttr {
TypeSourceInfo * typeHint;

public:
  // Factory methods
  static VecTypeHintAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * TypeHint, const AttributeCommonInfo &CommonInfo);
  static VecTypeHintAttr *Create(ASTContext &Ctx, TypeSourceInfo * TypeHint, const AttributeCommonInfo &CommonInfo);
  static VecTypeHintAttr *CreateImplicit(ASTContext &Ctx, TypeSourceInfo * TypeHint, SourceRange Range = {});
  static VecTypeHintAttr *Create(ASTContext &Ctx, TypeSourceInfo * TypeHint, SourceRange Range = {});

  // Constructors
  VecTypeHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , TypeSourceInfo * TypeHint
             );

  VecTypeHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  QualType getTypeHint() const {
    return typeHint->getType();
  }  TypeSourceInfo * getTypeHintLoc() const {
    return typeHint;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::VecTypeHint; }
};

class CLANG_ABI VectorCallAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_vectorcall = 0,
    CXX11_clang_vectorcall = 1,
    C23_clang_vectorcall = 2,
    Keyword_vectorcall = 3,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static VectorCallAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static VectorCallAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static VectorCallAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_vectorcall);
  static VectorCallAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_vectorcall);

  // Constructors
  VectorCallAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  VectorCallAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::VectorCall; }
};

class CLANG_ABI VisibilityAttr : public InheritableAttr {
public:
  enum VisibilityType {
    Default,
    Hidden,
    Protected
  };
private:
  VisibilityAttr::VisibilityType visibility;

public:
  enum Spelling {
    GNU_visibility = 0,
    CXX11_gnu_visibility = 1,
    C23_gnu_visibility = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static VisibilityAttr *CreateImplicit(ASTContext &Ctx, VisibilityAttr::VisibilityType Visibility, const AttributeCommonInfo &CommonInfo);
  static VisibilityAttr *Create(ASTContext &Ctx, VisibilityAttr::VisibilityType Visibility, const AttributeCommonInfo &CommonInfo);
  static VisibilityAttr *CreateImplicit(ASTContext &Ctx, VisibilityAttr::VisibilityType Visibility, SourceRange Range = {}, Spelling S = GNU_visibility);
  static VisibilityAttr *Create(ASTContext &Ctx, VisibilityAttr::VisibilityType Visibility, SourceRange Range = {}, Spelling S = GNU_visibility);

  // Constructors
  VisibilityAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , VisibilityAttr::VisibilityType Visibility
             );

  VisibilityAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  VisibilityAttr::VisibilityType getVisibility() const {
    return visibility;
  }

  static bool ConvertStrToVisibilityType(StringRef Val, VisibilityAttr::VisibilityType &Out);
  static const char *ConvertVisibilityTypeToStr(VisibilityAttr::VisibilityType Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::Visibility; }
};

class CLANG_ABI WarnUnusedAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_warn_unused = 0,
    CXX11_gnu_warn_unused = 1,
    C23_gnu_warn_unused = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WarnUnusedAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WarnUnusedAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WarnUnusedAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_warn_unused);
  static WarnUnusedAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_warn_unused);

  // Constructors
  WarnUnusedAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WarnUnusedAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::WarnUnused; }
};

class CLANG_ABI WarnUnusedResultAttr : public InheritableAttr {
unsigned messageLength;
char *message;

public:
  enum Spelling {
    CXX11_nodiscard = 0,
    C23_nodiscard = 1,
    CXX11_clang_warn_unused_result = 2,
    GNU_warn_unused_result = 3,
    CXX11_gnu_warn_unused_result = 4,
    C23_gnu_warn_unused_result = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WarnUnusedResultAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static WarnUnusedResultAttr *Create(ASTContext &Ctx, llvm::StringRef Message, const AttributeCommonInfo &CommonInfo);
  static WarnUnusedResultAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = CXX11_nodiscard);
  static WarnUnusedResultAttr *Create(ASTContext &Ctx, llvm::StringRef Message, SourceRange Range = {}, Spelling S = CXX11_nodiscard);

  // Constructors
  WarnUnusedResultAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Message
             );
  WarnUnusedResultAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WarnUnusedResultAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  llvm::StringRef getMessage() const {
    return llvm::StringRef(message, messageLength);
  }
  unsigned getMessageLength() const {
    return messageLength;
  }
  void setMessage(ASTContext &C, llvm::StringRef S) {
    messageLength = S.size();
    this->message = new (C, 1) char [messageLength];
    if (!S.empty())
      std::memcpy(this->message, S.data(), messageLength);
  }


    // Check whether this the C++11 nodiscard version, even in non C++11
    // spellings.
    bool IsCXX11NoDiscard() const {
      return this->getSemanticSpelling() == CXX11_nodiscard;
    }
  

  static bool classof(const Attr *A) { return A->getKind() == attr::WarnUnusedResult; }
};

class CLANG_ABI WeakAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_weak = 0,
    CXX11_gnu_weak = 1,
    C23_gnu_weak = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WeakAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WeakAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WeakAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_weak);
  static WeakAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_weak);

  // Constructors
  WeakAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WeakAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::Weak; }
};

class CLANG_ABI WeakImportAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_weak_import = 0,
    CXX11_clang_weak_import = 1,
    C23_clang_weak_import = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WeakImportAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WeakImportAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WeakImportAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_weak_import);
  static WeakImportAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_weak_import);

  // Constructors
  WeakImportAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WeakImportAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::WeakImport; }
};

class CLANG_ABI WeakRefAttr : public InheritableAttr {
unsigned aliaseeLength;
char *aliasee;

public:
  enum Spelling {
    GNU_weakref = 0,
    CXX11_gnu_weakref = 1,
    C23_gnu_weakref = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WeakRefAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Aliasee, const AttributeCommonInfo &CommonInfo);
  static WeakRefAttr *Create(ASTContext &Ctx, llvm::StringRef Aliasee, const AttributeCommonInfo &CommonInfo);
  static WeakRefAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef Aliasee, SourceRange Range = {}, Spelling S = GNU_weakref);
  static WeakRefAttr *Create(ASTContext &Ctx, llvm::StringRef Aliasee, SourceRange Range = {}, Spelling S = GNU_weakref);

  // Constructors
  WeakRefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef Aliasee
             );
  WeakRefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WeakRefAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getAliasee() const {
    return llvm::StringRef(aliasee, aliaseeLength);
  }
  unsigned getAliaseeLength() const {
    return aliaseeLength;
  }
  void setAliasee(ASTContext &C, llvm::StringRef S) {
    aliaseeLength = S.size();
    this->aliasee = new (C, 1) char [aliaseeLength];
    if (!S.empty())
      std::memcpy(this->aliasee, S.data(), aliaseeLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::WeakRef; }
};

class CLANG_ABI WebAssemblyExportNameAttr : public InheritableAttr {
unsigned exportNameLength;
char *exportName;

public:
  enum Spelling {
    GNU_export_name = 0,
    CXX11_clang_export_name = 1,
    C23_clang_export_name = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WebAssemblyExportNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ExportName, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyExportNameAttr *Create(ASTContext &Ctx, llvm::StringRef ExportName, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyExportNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ExportName, SourceRange Range = {}, Spelling S = GNU_export_name);
  static WebAssemblyExportNameAttr *Create(ASTContext &Ctx, llvm::StringRef ExportName, SourceRange Range = {}, Spelling S = GNU_export_name);

  // Constructors
  WebAssemblyExportNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef ExportName
             );

  WebAssemblyExportNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getExportName() const {
    return llvm::StringRef(exportName, exportNameLength);
  }
  unsigned getExportNameLength() const {
    return exportNameLength;
  }
  void setExportName(ASTContext &C, llvm::StringRef S) {
    exportNameLength = S.size();
    this->exportName = new (C, 1) char [exportNameLength];
    if (!S.empty())
      std::memcpy(this->exportName, S.data(), exportNameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::WebAssemblyExportName; }
};

class CLANG_ABI WebAssemblyFuncrefAttr : public TypeAttr {
public:
  // Factory methods
  static WebAssemblyFuncrefAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyFuncrefAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyFuncrefAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {});
  static WebAssemblyFuncrefAttr *Create(ASTContext &Ctx, SourceRange Range = {});

  // Constructors
  WebAssemblyFuncrefAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  WebAssemblyFuncrefAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::WebAssemblyFuncref; }
};

class CLANG_ABI WebAssemblyImportModuleAttr : public InheritableAttr {
unsigned importModuleLength;
char *importModule;

public:
  enum Spelling {
    GNU_import_module = 0,
    CXX11_clang_import_module = 1,
    C23_clang_import_module = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WebAssemblyImportModuleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ImportModule, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyImportModuleAttr *Create(ASTContext &Ctx, llvm::StringRef ImportModule, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyImportModuleAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ImportModule, SourceRange Range = {}, Spelling S = GNU_import_module);
  static WebAssemblyImportModuleAttr *Create(ASTContext &Ctx, llvm::StringRef ImportModule, SourceRange Range = {}, Spelling S = GNU_import_module);

  // Constructors
  WebAssemblyImportModuleAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef ImportModule
             );

  WebAssemblyImportModuleAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getImportModule() const {
    return llvm::StringRef(importModule, importModuleLength);
  }
  unsigned getImportModuleLength() const {
    return importModuleLength;
  }
  void setImportModule(ASTContext &C, llvm::StringRef S) {
    importModuleLength = S.size();
    this->importModule = new (C, 1) char [importModuleLength];
    if (!S.empty())
      std::memcpy(this->importModule, S.data(), importModuleLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::WebAssemblyImportModule; }
};

class CLANG_ABI WebAssemblyImportNameAttr : public InheritableAttr {
unsigned importNameLength;
char *importName;

public:
  enum Spelling {
    GNU_import_name = 0,
    CXX11_clang_import_name = 1,
    C23_clang_import_name = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static WebAssemblyImportNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ImportName, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyImportNameAttr *Create(ASTContext &Ctx, llvm::StringRef ImportName, const AttributeCommonInfo &CommonInfo);
  static WebAssemblyImportNameAttr *CreateImplicit(ASTContext &Ctx, llvm::StringRef ImportName, SourceRange Range = {}, Spelling S = GNU_import_name);
  static WebAssemblyImportNameAttr *Create(ASTContext &Ctx, llvm::StringRef ImportName, SourceRange Range = {}, Spelling S = GNU_import_name);

  // Constructors
  WebAssemblyImportNameAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , llvm::StringRef ImportName
             );

  WebAssemblyImportNameAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  llvm::StringRef getImportName() const {
    return llvm::StringRef(importName, importNameLength);
  }
  unsigned getImportNameLength() const {
    return importNameLength;
  }
  void setImportName(ASTContext &C, llvm::StringRef S) {
    importNameLength = S.size();
    this->importName = new (C, 1) char [importNameLength];
    if (!S.empty())
      std::memcpy(this->importName, S.data(), importNameLength);
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::WebAssemblyImportName; }
};

class CLANG_ABI WorkGroupSizeHintAttr : public InheritableAttr {
Expr * xDim;

Expr * yDim;

Expr * zDim;

public:
  // Factory methods
  static WorkGroupSizeHintAttr *CreateImplicit(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, const AttributeCommonInfo &CommonInfo);
  static WorkGroupSizeHintAttr *Create(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, const AttributeCommonInfo &CommonInfo);
  static WorkGroupSizeHintAttr *CreateImplicit(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, SourceRange Range = {});
  static WorkGroupSizeHintAttr *Create(ASTContext &Ctx, Expr * XDim, Expr * YDim, Expr * ZDim, SourceRange Range = {});

  // Constructors
  WorkGroupSizeHintAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , Expr * XDim
              , Expr * YDim
              , Expr * ZDim
             );

  WorkGroupSizeHintAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Expr * getXDim() const {
    return xDim;
  }

  Expr * getYDim() const {
    return yDim;
  }

  Expr * getZDim() const {
    return zDim;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::WorkGroupSizeHint; }
};

class CLANG_ABI X86ForceAlignArgPointerAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_force_align_arg_pointer = 0,
    CXX11_gnu_force_align_arg_pointer = 1,
    C23_gnu_force_align_arg_pointer = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static X86ForceAlignArgPointerAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static X86ForceAlignArgPointerAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static X86ForceAlignArgPointerAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_force_align_arg_pointer);
  static X86ForceAlignArgPointerAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_force_align_arg_pointer);

  // Constructors
  X86ForceAlignArgPointerAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  X86ForceAlignArgPointerAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;


  static bool classof(const Attr *A) { return A->getKind() == attr::X86ForceAlignArgPointer; }
};

class CLANG_ABI XRayInstrumentAttr : public InheritableAttr {
public:
  enum Spelling {
    GNU_xray_always_instrument = 0,
    CXX11_clang_xray_always_instrument = 1,
    C23_clang_xray_always_instrument = 2,
    GNU_xray_never_instrument = 3,
    CXX11_clang_xray_never_instrument = 4,
    C23_clang_xray_never_instrument = 5,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static XRayInstrumentAttr *CreateImplicit(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static XRayInstrumentAttr *Create(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo);
  static XRayInstrumentAttr *CreateImplicit(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_xray_always_instrument);
  static XRayInstrumentAttr *Create(ASTContext &Ctx, SourceRange Range = {}, Spelling S = GNU_xray_always_instrument);

  // Constructors
  XRayInstrumentAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
             );

  XRayInstrumentAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  Spelling getSemanticSpelling() const;
  bool alwaysXRayInstrument() const { return getAttributeSpellingListIndex() == 0 ||
    getAttributeSpellingListIndex() == 1 ||
    getAttributeSpellingListIndex() == 2; }
  bool neverXRayInstrument() const { return getAttributeSpellingListIndex() == 3 ||
    getAttributeSpellingListIndex() == 4 ||
    getAttributeSpellingListIndex() == 5; }


  static bool classof(const Attr *A) { return A->getKind() == attr::XRayInstrument; }
};

class CLANG_ABI XRayLogArgsAttr : public InheritableAttr {
unsigned argumentCount;

public:
  enum Spelling {
    GNU_xray_log_args = 0,
    CXX11_clang_xray_log_args = 1,
    C23_clang_xray_log_args = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static XRayLogArgsAttr *CreateImplicit(ASTContext &Ctx, unsigned ArgumentCount, const AttributeCommonInfo &CommonInfo);
  static XRayLogArgsAttr *Create(ASTContext &Ctx, unsigned ArgumentCount, const AttributeCommonInfo &CommonInfo);
  static XRayLogArgsAttr *CreateImplicit(ASTContext &Ctx, unsigned ArgumentCount, SourceRange Range = {}, Spelling S = GNU_xray_log_args);
  static XRayLogArgsAttr *Create(ASTContext &Ctx, unsigned ArgumentCount, SourceRange Range = {}, Spelling S = GNU_xray_log_args);

  // Constructors
  XRayLogArgsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , unsigned ArgumentCount
             );

  XRayLogArgsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  unsigned getArgumentCount() const {
    return argumentCount;
  }



  static bool classof(const Attr *A) { return A->getKind() == attr::XRayLogArgs; }
};

class CLANG_ABI ZeroCallUsedRegsAttr : public InheritableAttr {
public:
  enum ZeroCallUsedRegsKind {
    Skip,
    UsedGPRArg,
    UsedGPR,
    UsedArg,
    Used,
    AllGPRArg,
    AllGPR,
    AllArg,
    All
  };
private:
  ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind zeroCallUsedRegs;

public:
  enum Spelling {
    GNU_zero_call_used_regs = 0,
    CXX11_gnu_zero_call_used_regs = 1,
    C23_gnu_zero_call_used_regs = 2,
  SpellingNotCalculated = 15

  };

  // Factory methods
  static ZeroCallUsedRegsAttr *CreateImplicit(ASTContext &Ctx, ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind ZeroCallUsedRegs, const AttributeCommonInfo &CommonInfo);
  static ZeroCallUsedRegsAttr *Create(ASTContext &Ctx, ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind ZeroCallUsedRegs, const AttributeCommonInfo &CommonInfo);
  static ZeroCallUsedRegsAttr *CreateImplicit(ASTContext &Ctx, ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind ZeroCallUsedRegs, SourceRange Range = {}, Spelling S = GNU_zero_call_used_regs);
  static ZeroCallUsedRegsAttr *Create(ASTContext &Ctx, ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind ZeroCallUsedRegs, SourceRange Range = {}, Spelling S = GNU_zero_call_used_regs);

  // Constructors
  ZeroCallUsedRegsAttr(ASTContext &Ctx, const AttributeCommonInfo &CommonInfo
              , ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind ZeroCallUsedRegs
             );

  ZeroCallUsedRegsAttr *clone(ASTContext &C) const;
  void printPretty(raw_ostream &OS,
                   const PrintingPolicy &Policy) const;
  const char *getSpelling() const;
  ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind getZeroCallUsedRegs() const {
    return zeroCallUsedRegs;
  }

  static bool ConvertStrToZeroCallUsedRegsKind(StringRef Val, ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind &Out);
  static const char *ConvertZeroCallUsedRegsKindToStr(ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind Val);


  static bool classof(const Attr *A) { return A->getKind() == attr::ZeroCallUsedRegs; }
};

#endif // LLVM_CLANG_ATTR_CLASSES_INC
