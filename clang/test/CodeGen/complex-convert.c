// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - %s | FileCheck %s

// Test conversions between complex integer types and standard integer
// types.  Tests binary operator conversion and assignment conversion
// with widening, narrowing, and equal-size operands.  Signed and unsigned
// variations.  Attempts to work for all targets.  Assumptions:
//
//  * "char" and "long long" are of different lengths (CHSIZE and LLSIZE).
//  * Arithmetic is not performed directly on "char" type.

void foo(signed char sc, unsigned char uc, signed long long sll,
         unsigned long long ull, _Complex signed char csc,
         _Complex unsigned char cuc, _Complex signed long long csll,
         _Complex unsigned long long cull) {

  signed char sc1;
  unsigned char uc1;
  signed long long sll1;
  unsigned long long ull1;
  _Complex signed char csc1;
  _Complex unsigned char cuc1;
  _Complex signed long long csll1;
  _Complex unsigned long long cull1;
  // CHECK-LABEL: define {{.*}}void @foo(
  // Match the prototype to pick up the size of sc and sll.
  // CHECK: i[[CHSIZE:[0-9]+]]{{[^,]*}},
  // CHECK: i[[CHSIZE]]{{[^,]*}},
  // CHECK: i[[LLSIZE:[0-9]+]]

  // Match against the allocas to pick up the alignments.
  // CHECK: alloca i[[CHSIZE]], align [[CHALIGN:[0-9]+]]
  // CHECK: alloca i[[LLSIZE]], align [[LLALIGN:[0-9]+]]

  // CHECK: store i64 %ull,

  sc1 = csc;
  // CHECK: %[[VAR1:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CSC:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR2:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR1]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR2]], i[[CHSIZE]]* %[[SC1:[A-Za-z0-9.]+]], align [[CHALIGN]]

  sc1 = cuc;
  // CHECK-NEXT: %[[VAR3:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CUC:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR4:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR3]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR4]], i[[CHSIZE]]* %[[SC1]], align [[CHALIGN]]

  sc1 = csll;
  // CHECK-NEXT: %[[VAR5:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]]  }, { i[[LLSIZE]], i[[LLSIZE]]  }* %[[CSLL:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR6:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR5]]
  // CHECK-NEXT: %[[VAR7:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR6]] to i[[CHSIZE]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR7]], i[[CHSIZE]]* %[[SC1]], align [[CHALIGN]]

  sc1 = cull;
  // CHECK-NEXT: %[[VAR8:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]]  }, { i[[LLSIZE]], i[[LLSIZE]]  }* %[[CULL:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR9:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR8]]
  // CHECK-NEXT: %[[VAR10:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR9]] to i[[CHSIZE]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR10]], i[[CHSIZE]]* %[[SC1]], align [[CHALIGN]]
  
  uc1 = csc;
  // CHECK-NEXT: %[[VAR11:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR12:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR11]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR12]], i[[CHSIZE]]* %[[UC1:[A-Za-z0-9.]+]], align [[CHALIGN]]

  uc1 = cuc;
  // CHECK-NEXT: %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR14:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR13]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR14]], i[[CHSIZE]]* %[[UC1]], align [[CHALIGN]]

  uc1 = csll;
  // CHECK-NEXT: %[[VAR15:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]]  }, { i[[LLSIZE]], i[[LLSIZE]]  }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR16:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR15]]
  // CHECK-NEXT: %[[VAR17:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR16]] to i[[CHSIZE]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR17]], i[[CHSIZE]]* %[[UC1]], align [[CHALIGN]]

  uc1 = cull;
  // CHECK-NEXT: %[[VAR18:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]]  }, { i[[LLSIZE]], i[[LLSIZE]]  }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR19:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR18]]
  // CHECK-NEXT: %[[VAR20:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR19]] to i[[CHSIZE]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR20]], i[[CHSIZE]]* %[[UC1]], align [[CHALIGN]]

  sll1 = csc;
  // CHECK-NEXT: %[[VAR21:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR22:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR21]]
  // CHECK-NEXT: %[[VAR23:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR22]] to i[[LLSIZE]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR23]], i[[LLSIZE]]* %[[SLL1:[A-Za-z0-9]+]], align [[LLALIGN]]

  sll1 = cuc;
  // CHECK-NEXT: %[[VAR24:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]]  }, { i[[CHSIZE]], i[[CHSIZE]]  }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR25:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR24]]
  // CHECK-NEXT: %[[VAR26:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR25]] to i[[LLSIZE]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR26]], i[[LLSIZE]]* %[[SLL1]], align [[LLALIGN]]

  sll1 = csll;
  // CHECK-NEXT: %[[VAR27:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR28:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR27]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR28]], i[[LLSIZE]]* %[[SLL1]], align [[LLALIGN]]

  sll1 = cull;
  // CHECK-NEXT: %[[VAR29:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR30:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR29]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR30]], i[[LLSIZE]]* %[[SLL1]], align [[LLALIGN]]
  
  ull1 = csc;
  // CHECK-NEXT: %[[VAR31:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR32:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR31]]
  // CHECK-NEXT: %[[VAR33:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR32]] to i[[LLSIZE]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR33]], i[[LLSIZE]]* %[[ULL1:[A-Za-z0-9]+]], align [[LLALIGN]]

  ull1 = cuc;
  // CHECK-NEXT: %[[VAR34:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR35:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR34]]
  // CHECK-NEXT: %[[VAR36:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR35]] to i[[LLSIZE]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR36]], i[[LLSIZE]]* %[[ULL1]], align [[LLALIGN]]

  ull1 = csll;
  // CHECK-NEXT: %[[VAR37:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR38:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR37]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR38]], i[[LLSIZE]]* %[[ULL1]], align [[LLALIGN]]

  ull1 = cull;
  // CHECK-NEXT: %[[VAR39:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR40:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR39]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR40]], i[[LLSIZE]]* %[[ULL1]], align [[LLALIGN]]

  csc1 = sc;
  // CHECK-NEXT: %[[VAR41:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR:[A-Za-z0-9.]+]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR42:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR43:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR41]], i[[CHSIZE]]* %[[VAR42]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR43]]

  csc1 = uc;
  // CHECK-NEXT: %[[VAR44:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR:[A-Za-z0-9.]+]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR45:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR46:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR44]], i[[CHSIZE]]* %[[VAR45]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR46]]

  csc1 = sll;
  // CHECK-NEXT: %[[VAR47:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR:[A-Za-z0-9.]+]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR48:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR47]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR49:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR50:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR48]], i[[CHSIZE]]* %[[VAR49]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR50]]

  csc1 = ull;
  // CHECK-NEXT: %[[VAR51:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR:[A-Za-z0-9.]+]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR52:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR51]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR53:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR54:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR52]], i[[CHSIZE]]* %[[VAR53]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR54]]
  
  cuc1 = sc;
  // CHECK-NEXT: %[[VAR55:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR56:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR57:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR55]], i[[CHSIZE]]* %[[VAR56]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR57]]

  cuc1 = uc;
  // CHECK-NEXT: %[[VAR58:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR59:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR60:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR58]], i[[CHSIZE]]* %[[VAR59]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR60]]

  cuc1 = sll;
  // CHECK-NEXT: %[[VAR61:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR62:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR61]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR63:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR64:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR62]], i[[CHSIZE]]* %[[VAR63]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR64]]

  cuc1 = ull;
  // CHECK-NEXT: %[[VAR65:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR66:[A-Za-z0-9.]+]] = trunc i[[LLSIZE]] %[[VAR65]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR67:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR68:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR66]], i[[CHSIZE]]* %[[VAR67]]
  // CHECK-NEXT: store i[[CHSIZE]] 0, i[[CHSIZE]]* %[[VAR68]]

  csll1 = sc;
  // CHECK-NEXT: %[[VAR69:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR70:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR69]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR71:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR72:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR70]], i[[LLSIZE]]* %[[VAR71]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR72]]

  csll1 = uc;
  // CHECK-NEXT: %[[VAR73:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR74:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR73]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR75:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR76:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR74]], i[[LLSIZE]]* %[[VAR75]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR76]]

  csll1 = sll;
  // CHECK-NEXT: %[[VAR77:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR78:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR79:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR77]], i[[LLSIZE]]* %[[VAR78]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR79]]

  csll1 = ull;
  // CHECK-NEXT: %[[VAR77:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR78:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR79:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR77]], i[[LLSIZE]]* %[[VAR78]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR79]]

  cull1 = sc;
  // CHECK-NEXT: %[[VAR80:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR81:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR80]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR82:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1:[A-Za-z0-9.]+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR83:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR81]], i[[LLSIZE]]* %[[VAR82]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR83]]

  cull1 = uc;
  // CHECK-NEXT: %[[VAR84:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR85:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR84]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR86:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR87:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR85]], i[[LLSIZE]]* %[[VAR86]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR87]]

  cull1 = sll;
  // CHECK-NEXT: %[[VAR88:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR89:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR90:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR88]], i[[LLSIZE]]* %[[VAR89]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR90]]

  cull1 = ull;
  // CHECK-NEXT: %[[VAR91:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR92:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR93:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR91]], i[[LLSIZE]]* %[[VAR92]]
  // CHECK-NEXT: store i[[LLSIZE]] 0, i[[LLSIZE]]* %[[VAR93]]

  csc1 = sc + csc;
  // CHECK-NEXT: %[[VAR94:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR95:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR94]] to i[[ARSIZE:[0-9]+]]
  // CHECK-NEXT: %[[VAR96:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR97:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR96]]
  // CHECK-NEXT: %[[VAR98:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR99:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR98]]
  // CHECK-NEXT: %[[VAR100:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR97]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR101:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR99]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR102:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR95]], %[[VAR100]]
  // CHECK-NEXT: %[[VAR103:[A-Za-z0-9.]+]] = add i[[ARSIZE]] 0, %[[VAR101]]
  // CHECK-NEXT: %[[VAR104:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR102]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR105:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR103]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR106:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR107:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR104]], i[[CHSIZE]]* %[[VAR106]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR105]], i[[CHSIZE]]* %[[VAR107]]

  cuc1 = sc + cuc;
  // CHECK-NEXT: %[[VAR108:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR109:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR108]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR110:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR111:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR110]]
  // CHECK-NEXT: %[[VAR112:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR113:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR112]]
  // CHECK-NEXT: %[[VAR114:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR111]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR115:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR113]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR116:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR109]], %[[VAR114]]
  // CHECK-NEXT: %[[VAR117:[A-Za-z0-9.]+]] = add i[[ARSIZE]] 0, %[[VAR115]]
  // CHECK-NEXT: %[[VAR118:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR116]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR119:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR117]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR120:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR121:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR118]], i[[CHSIZE]]* %[[VAR120]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR119]], i[[CHSIZE]]* %[[VAR121]]

  csll1 = sc + csll;
  // CHECK-NEXT: %[[VAR122:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR123:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR122]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR124:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR125:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR124]]
  // CHECK-NEXT: %[[VAR126:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR127:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR126]]
  // CHECK-NEXT: %[[VAR128:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR123]], %[[VAR125]]
  // CHECK-NEXT: %[[VAR129:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR127]]
  // CHECK-NEXT: %[[VAR130:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR131:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR128]], i[[LLSIZE]]* %[[VAR130]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR129]], i[[LLSIZE]]* %[[VAR131]]

  cull1 = sc + cull;
  // CHECK-NEXT: %[[VAR132:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR133:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR132]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR134:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR135:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR134]]
  // CHECK-NEXT: %[[VAR136:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR137:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR136]]
  // CHECK-NEXT: %[[VAR138:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR133]], %[[VAR135]]
  // CHECK-NEXT: %[[VAR139:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR137]]
  // CHECK-NEXT: %[[VAR140:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR141:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR138]], i[[LLSIZE]]* %[[VAR140]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR139]], i[[LLSIZE]]* %[[VAR141]]
  
  csc1 = uc + csc;
  // CHECK-NEXT: %[[VAR142:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR143:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR142]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR144:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR145:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR144]]
  // CHECK-NEXT: %[[VAR146:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR147:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR146]]
  // CHECK-NEXT: %[[VAR148:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR145]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR149:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR147]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR150:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR143]], %[[VAR148]]
  // CHECK-NEXT: %[[VAR151:[A-Za-z0-9.]+]] = add i[[ARSIZE]] 0, %[[VAR149]]
  // CHECK-NEXT: %[[VAR152:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR150]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR153:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR151]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR154:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR155:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR152]], i[[CHSIZE]]* %[[VAR154]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR153]], i[[CHSIZE]]* %[[VAR155]]

  cuc1 = uc + cuc;
  // CHECK-NEXT: %[[VAR156:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR157:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR156]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR158:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR159:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR158]]
  // CHECK-NEXT: %[[VAR160:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR161:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR160]]
  // CHECK-NEXT: %[[VAR162:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR159]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR163:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR161]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR164:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR157]], %[[VAR162]]
  // CHECK-NEXT: %[[VAR165:[A-Za-z0-9.]+]] = add i[[ARSIZE]] 0, %[[VAR163]]
  // CHECK-NEXT: %[[VAR166:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR164]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR167:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR165]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR168:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR169:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR166]], i[[CHSIZE]]* %[[VAR168]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR167]], i[[CHSIZE]]* %[[VAR169]]

  csll1 = uc + csll;
  // CHECK-NEXT: %[[VAR170:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR171:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR170]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR172:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR173:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR172]]
  // CHECK-NEXT: %[[VAR174:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR175:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR174]]
  // CHECK-NEXT: %[[VAR176:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR171]], %[[VAR173]]
  // CHECK-NEXT: %[[VAR177:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR175]]
  // CHECK-NEXT: %[[VAR178:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR179:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR176]], i[[LLSIZE]]* %[[VAR178]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR177]], i[[LLSIZE]]* %[[VAR179]]

  cull1 = uc + cull;
  // CHECK-NEXT: %[[VAR180:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR181:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR180]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR182:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR183:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR182]]
  // CHECK-NEXT: %[[VAR184:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR185:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR184]]
  // CHECK-NEXT: %[[VAR186:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR181]], %[[VAR183]]
  // CHECK-NEXT: %[[VAR187:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR185]]
  // CHECK-NEXT: %[[VAR188:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR189:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR186]], i[[LLSIZE]]* %[[VAR188]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR187]], i[[LLSIZE]]* %[[VAR189]]

  csll1 = sll + csc;
  // CHECK-NEXT: %[[VAR190:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR191:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR192:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR191]]
  // CHECK-NEXT: %[[VAR193:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR194:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR193]]
  // CHECK-NEXT: %[[VAR195:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR192]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR196:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR194]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR197:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR190]], %[[VAR195]]
  // CHECK-NEXT: %[[VAR198:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR196]]
  // CHECK-NEXT: %[[VAR199:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR200:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR197]], i[[LLSIZE]]* %[[VAR199]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR198]], i[[LLSIZE]]* %[[VAR200]]

  csll1 = sll + cuc;
  // CHECK-NEXT: %[[VAR201:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR202:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR203:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR202]]
  // CHECK-NEXT: %[[VAR204:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR205:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR204]]
  // CHECK-NEXT: %[[VAR206:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR203]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR207:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR205]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR208:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR201]], %[[VAR206]]
  // CHECK-NEXT: %[[VAR209:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR207]]
  // CHECK-NEXT: %[[VAR210:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR211:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR208]], i[[LLSIZE]]* %[[VAR210]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR209]], i[[LLSIZE]]* %[[VAR211]]

  csll1 = sll + csll;
  // CHECK-NEXT: %[[VAR212:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR213:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR214:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR213]]
  // CHECK-NEXT: %[[VAR215:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR216:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR215]]
  // CHECK-NEXT: %[[VAR217:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR212]], %[[VAR214]]
  // CHECK-NEXT: %[[VAR218:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR216]]
  // CHECK-NEXT: %[[VAR219:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR220:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR217]], i[[LLSIZE]]* %[[VAR219]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR218]], i[[LLSIZE]]* %[[VAR220]]

  csll1 = sll + cull;
  // CHECK-NEXT: %[[VAR221:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR222:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR223:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR222]]
  // CHECK-NEXT: %[[VAR224:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR225:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR224]]
  // CHECK-NEXT: %[[VAR226:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR221]], %[[VAR223]]
  // CHECK-NEXT: %[[VAR227:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR225]]
  // CHECK-NEXT: %[[VAR228:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR229:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR226]], i[[LLSIZE]]* %[[VAR228]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR227]], i[[LLSIZE]]* %[[VAR229]]
  
  csll1 = ull + csc;
  // CHECK-NEXT: %[[VAR230:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR231:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR232:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR231]]
  // CHECK-NEXT: %[[VAR233:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR234:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR233]]
  // CHECK-NEXT: %[[VAR235:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR232]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR236:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR234]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR237:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR230]], %[[VAR235]]
  // CHECK-NEXT: %[[VAR238:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR236]]
  // CHECK-NEXT: %[[VAR239:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR240:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR237]], i[[LLSIZE]]* %[[VAR239]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR238]], i[[LLSIZE]]* %[[VAR240]]

  cull1 = ull + cuc;
  // CHECK-NEXT: %[[VAR241:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR242:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR243:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR242]]
  // CHECK-NEXT: %[[VAR244:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR245:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR244]]
  // CHECK-NEXT: %[[VAR246:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR243]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR247:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR245]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR248:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR241]], %[[VAR246]]
  // CHECK-NEXT: %[[VAR249:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR247]]
  // CHECK-NEXT: %[[VAR250:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR251:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR248]], i[[LLSIZE]]* %[[VAR250]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR249]], i[[LLSIZE]]* %[[VAR251]]

  csll1 = ull + csll;
  // CHECK-NEXT: %[[VAR252:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR253:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR254:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR253]]
  // CHECK-NEXT: %[[VAR255:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR256:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR255]]
  // CHECK-NEXT: %[[VAR257:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR252]], %[[VAR254]]
  // CHECK-NEXT: %[[VAR258:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR256]]
  // CHECK-NEXT: %[[VAR259:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR260:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR257]], i[[LLSIZE]]* %[[VAR259]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR258]], i[[LLSIZE]]* %[[VAR260]]

  cull1 = ull + cull;
  // CHECK-NEXT: %[[VAR261:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR262:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR263:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR262]]
  // CHECK-NEXT: %[[VAR264:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR265:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR264]]
  // CHECK-NEXT: %[[VAR266:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR261]], %[[VAR263]]
  // CHECK-NEXT: %[[VAR267:[A-Za-z0-9.]+]] = add i[[LLSIZE]] 0, %[[VAR265]]
  // CHECK-NEXT: %[[VAR268:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR269:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR266]], i[[LLSIZE]]* %[[VAR268]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR267]], i[[LLSIZE]]* %[[VAR269]]

  csc1 = csc + sc;
  // CHECK-NEXT: %[[VAR270:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR271:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR270]]
  // CHECK-NEXT: %[[VAR272:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR273:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR272]]
  // CHECK-NEXT: %[[VAR274:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR271]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR275:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR273]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR276:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR277:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR276]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR278:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR274]], %[[VAR277]]
  // CHECK-NEXT: %[[VAR279:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR275]], 0
  // CHECK-NEXT: %[[VAR280:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR278]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR281:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR279]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR282:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR283:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR280]], i[[CHSIZE]]* %[[VAR282]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR281]], i[[CHSIZE]]* %[[VAR283]]

  csc1 = csc + uc;
  // CHECK-NEXT: %[[VAR284:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR285:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR284]]
  // CHECK-NEXT: %[[VAR286:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR287:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR286]]
  // CHECK-NEXT: %[[VAR288:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR285]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR289:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR287]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR290:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR291:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR290]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR292:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR288]], %[[VAR291]]
  // CHECK-NEXT: %[[VAR293:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR289]], 0
  // CHECK-NEXT: %[[VAR294:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR292]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR295:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR293]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR296:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR297:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR294]], i[[CHSIZE]]* %[[VAR296]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR295]], i[[CHSIZE]]* %[[VAR297]]

  csll1 = csc + sll;
  // CHECK-NEXT: %[[VAR298:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR299:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR298]]
  // CHECK-NEXT: %[[VAR300:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR301:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR300]]
  // CHECK-NEXT: %[[VAR302:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR299]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR303:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR301]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR304:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR305:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR302]], %[[VAR304]]
  // CHECK-NEXT: %[[VAR306:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR303]], 0
  // CHECK-NEXT: %[[VAR307:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR308:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR305]], i[[LLSIZE]]* %[[VAR307]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR306]], i[[LLSIZE]]* %[[VAR308]]

  csll1 = csc + ull;
  // CHECK-NEXT: %[[VAR309:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR310:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR309]]
  // CHECK-NEXT: %[[VAR311:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR312:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR311]]
  // CHECK-NEXT: %[[VAR313:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR310]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR314:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR312]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR315:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR316:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR313]], %[[VAR315]]
  // CHECK-NEXT: %[[VAR317:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR314]], 0
  // CHECK-NEXT: %[[VAR318:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR319:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR316]], i[[LLSIZE]]* %[[VAR318]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR317]], i[[LLSIZE]]* %[[VAR319]]
  
  csc1 = cuc + sc;
  // CHECK-NEXT: %[[VAR320:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR321:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR320]]
  // CHECK-NEXT: %[[VAR322:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR323:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR322]]
  // CHECK-NEXT: %[[VAR324:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR321]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR325:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR323]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR326:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR327:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR326]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR328:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR324]], %[[VAR327]]
  // CHECK-NEXT: %[[VAR329:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR325]], 0
  // CHECK-NEXT: %[[VAR330:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR328]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR331:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR329]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR332:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR333:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CSC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR330]], i[[CHSIZE]]* %[[VAR332]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR331]], i[[CHSIZE]]* %[[VAR333]]

  cuc1 = cuc + uc;
  // CHECK-NEXT: %[[VAR334:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR335:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR334]]
  // CHECK-NEXT: %[[VAR336:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR337:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR336]]
  // CHECK-NEXT: %[[VAR338:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR335]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR339:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR337]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR340:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR341:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR340]] to i[[ARSIZE]]
  // CHECK-NEXT: %[[VAR342:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR338]], %[[VAR341]]
  // CHECK-NEXT: %[[VAR343:[A-Za-z0-9.]+]] = add i[[ARSIZE]] %[[VAR339]], 0
  // CHECK-NEXT: %[[VAR344:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR342]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR345:[A-Za-z0-9.]+]] = trunc i[[ARSIZE]] %[[VAR343]] to i[[CHSIZE]]
  // CHECK-NEXT: %[[VAR346:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR347:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR344]], i[[CHSIZE]]* %[[VAR346]]
  // CHECK-NEXT: store i[[CHSIZE]] %[[VAR345]], i[[CHSIZE]]* %[[VAR347]]

  csll1 = cuc + sll;
  // CHECK-NEXT: %[[VAR348:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR349:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR348]]
  // CHECK-NEXT: %[[VAR350:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR351:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR350]]
  // CHECK-NEXT: %[[VAR352:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR349]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR353:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR351]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR354:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR355:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR352]], %[[VAR354]]
  // CHECK-NEXT: %[[VAR356:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR353]], 0
  // CHECK-NEXT: %[[VAR357:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR358:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR355]], i[[LLSIZE]]* %[[VAR357]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR356]], i[[LLSIZE]]* %[[VAR358]]

  cull1 = cuc + ull;
  // CHECK-NEXT: %[[VAR357:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR358:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR357]]
  // CHECK-NEXT: %[[VAR359:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[CHSIZE]], i[[CHSIZE]] }, { i[[CHSIZE]], i[[CHSIZE]] }* %[[CUC]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR360:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[VAR359]]
  // CHECK-NEXT: %[[VAR361:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR358]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR362:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR360]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR363:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR364:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR361]], %[[VAR363]]
  // CHECK-NEXT: %[[VAR365:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR362]], 0
  // CHECK-NEXT: %[[VAR366:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR367:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR364]], i[[LLSIZE]]* %[[VAR366]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR365]], i[[LLSIZE]]* %[[VAR367]]

  csll1 = csll + sc;
  // CHECK-NEXT: %[[VAR368:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR369:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR368]]
  // CHECK-NEXT: %[[VAR370:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR371:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR370]]
  // CHECK-NEXT: %[[VAR372:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR373:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR372]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR374:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR369]], %[[VAR373]]
  // CHECK-NEXT: %[[VAR375:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR371]], 0
  // CHECK-NEXT: %[[VAR376:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR377:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR374]], i[[LLSIZE]]* %[[VAR376]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR375]], i[[LLSIZE]]* %[[VAR377]]

  csll1 = csll + uc;
  // CHECK-NEXT: %[[VAR378:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR379:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR378]]
  // CHECK-NEXT: %[[VAR380:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR381:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR380]]
  // CHECK-NEXT: %[[VAR382:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR383:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR382]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR384:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR379]], %[[VAR383]]
  // CHECK-NEXT: %[[VAR385:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR381]], 0
  // CHECK-NEXT: %[[VAR386:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR387:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR384]], i[[LLSIZE]]* %[[VAR386]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR385]], i[[LLSIZE]]* %[[VAR387]]

  csll1 = csll + sll;
  // CHECK-NEXT: %[[VAR388:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR389:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR388]]
  // CHECK-NEXT: %[[VAR390:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR391:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR390]]
  // CHECK-NEXT: %[[VAR392:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR393:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR389]], %[[VAR392]]
  // CHECK-NEXT: %[[VAR394:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR391]], 0
  // CHECK-NEXT: %[[VAR395:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR396:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR393]], i[[LLSIZE]]* %[[VAR395]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR394]], i[[LLSIZE]]* %[[VAR396]]

  csll1 = csll + ull;
  // CHECK-NEXT: %[[VAR397:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR398:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR397]]
  // CHECK-NEXT: %[[VAR399:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR400:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR399]]
  // CHECK-NEXT: %[[VAR401:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR402:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR398]], %[[VAR401]]
  // CHECK-NEXT: %[[VAR403:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR400]], 0
  // CHECK-NEXT: %[[VAR404:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR405:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR402]], i[[LLSIZE]]* %[[VAR404]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR403]], i[[LLSIZE]]* %[[VAR405]]
  
  csll1 = cull + sc;
  // CHECK-NEXT: %[[VAR406:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR407:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR406]]
  // CHECK-NEXT: %[[VAR408:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR409:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR408]]
  // CHECK-NEXT: %[[VAR410:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[SCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR411:[A-Za-z0-9.]+]] = sext i[[CHSIZE]] %[[VAR410]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR412:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR407]], %[[VAR411]]
  // CHECK-NEXT: %[[VAR413:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR409]], 0
  // CHECK-NEXT: %[[VAR414:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR415:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR412]], i[[LLSIZE]]* %[[VAR414]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR413]], i[[LLSIZE]]* %[[VAR415]]

  cull1 = cull + uc;
  // CHECK-NEXT: %[[VAR416:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR417:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR416]]
  // CHECK-NEXT: %[[VAR418:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR419:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR418]]
  // CHECK-NEXT: %[[VAR420:[A-Za-z0-9.]+]] = load i[[CHSIZE]], i[[CHSIZE]]* %[[UCADDR]], align [[CHALIGN]]
  // CHECK-NEXT: %[[VAR421:[A-Za-z0-9.]+]] = zext i[[CHSIZE]] %[[VAR420]] to i[[LLSIZE]]
  // CHECK-NEXT: %[[VAR422:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR417]], %[[VAR421]]
  // CHECK-NEXT: %[[VAR423:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR419]], 0
  // CHECK-NEXT: %[[VAR424:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR425:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR422]], i[[LLSIZE]]* %[[VAR424]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR423]], i[[LLSIZE]]* %[[VAR425]]

  csll1 = cull + sll;
  // CHECK-NEXT: %[[VAR426:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR427:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR426]]
  // CHECK-NEXT: %[[VAR428:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR429:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR428]]
  // CHECK-NEXT: %[[VAR430:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[SLLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR431:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR427]], %[[VAR430]]
  // CHECK-NEXT: %[[VAR432:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR429]], 0
  // CHECK-NEXT: %[[VAR433:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR434:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CSLL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR431]], i[[LLSIZE]]* %[[VAR433]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR432]], i[[LLSIZE]]* %[[VAR434]]

  cull1 = cull + ull;
  // CHECK-NEXT: %[[VAR435:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR436:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR435]]
  // CHECK-NEXT: %[[VAR437:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: %[[VAR438:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[VAR437]]
  // CHECK-NEXT: %[[VAR439:[A-Za-z0-9.]+]] = load i[[LLSIZE]], i[[LLSIZE]]* %[[ULLADDR]], align [[LLALIGN]]
  // CHECK-NEXT: %[[VAR440:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR436]], %[[VAR439]]
  // CHECK-NEXT: %[[VAR441:[A-Za-z0-9.]+]] = add i[[LLSIZE]] %[[VAR438]], 0
  // CHECK-NEXT: %[[VAR442:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK-NEXT: %[[VAR443:[A-Za-z0-9.]+]] = getelementptr inbounds { i[[LLSIZE]], i[[LLSIZE]] }, { i[[LLSIZE]], i[[LLSIZE]] }* %[[CULL1]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR440]], i[[LLSIZE]]* %[[VAR442]]
  // CHECK-NEXT: store i[[LLSIZE]] %[[VAR441]], i[[LLSIZE]]* %[[VAR443]]
}

// This code used to cause a crash; test that it no longer does so.
_Complex int a;
void pr44624(void) {
  (_Complex double) a;
}
