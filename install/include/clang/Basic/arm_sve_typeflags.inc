#ifdef LLVM_GET_SVE_TYPEFLAGS
const uint64_t SplatOperandMask = 7168;
const uint64_t IsTupleCreate = 4294967296;
const uint64_t IsOutZA = 17592186044416;
const uint64_t IsWriteZA = 1099511627776;
const uint64_t IsInZA = 8796093022208;
const uint64_t IsStreaming = 137438953472;
const uint64_t IsReadZA = 549755813888;
const uint64_t IsInsertOp1SVALL = 67108864;
const uint64_t VerifyRuntimeMode = 4398046511104;
const uint64_t IsByteIndexed = 16777216;
const uint64_t IsOutZT0 = 140737488355328;
const uint64_t IsInZT0 = 70368744177664;
const uint64_t IsOverloadWhileRW = 4194304;
const uint64_t IsStructLoad = 131072;
const uint64_t IsTupleGet = 8589934592;
const uint64_t IsScatterStore = 65536;
const uint64_t IsAppendSVALL = 33554432;
const uint64_t IsOverloadCvt = 8388608;
const uint64_t IsGatherPrefetch = 268435456;
const uint64_t ReverseUSDOT = 1073741824;
const uint64_t IsLoad = 8192;
const uint64_t ReverseMergeAnyBinOp = 34359738368;
const uint64_t EltTypeMask = 15;
const uint64_t IsPrefetch = 134217728;
const uint64_t IsZExtReturn = 524288;
const uint64_t IsOverloadNone = 1048576;
const uint64_t SetsFPMR = 562949953421312;
const uint64_t FirstMemEltType = 16;
const uint64_t IsUndef = 2147483648;
const uint64_t MemEltTypeMask = 112;
const uint64_t IsInOutZA = 35184372088832;
const uint64_t IsInOutZT0 = 281474976710656;
const uint64_t FirstSplatOperand = 1024;
const uint64_t IsStore = 16384;
const uint64_t ReverseCompare = 536870912;
const uint64_t MergeTypeMask = 896;
const uint64_t ReverseMergeAnyAccOp = 68719476736;
const uint64_t IsOverloadWhileOrMultiVecCvt = 2097152;
const uint64_t IsReductionQV = 2199023255552;
const uint64_t FirstEltType = 1;
const uint64_t NoFlags = 0;
const uint64_t IsTupleSet = 17179869184;
const uint64_t IsStreamingCompatible = 274877906944;
const uint64_t FirstMergeTypeMask = 128;
const uint64_t IsStructStore = 262144;
const uint64_t IsGatherLoad = 32768;
const uint64_t OverloadKindMask = 14680064;
#endif

#ifdef LLVM_GET_SVE_ELTTYPES
  EltTyFloat64 = 8,
  EltTyInt128 = 5,
  EltTyFloat16 = 6,
  EltTyInt64 = 4,
  EltTyBool8 = 9,
  EltTyInt16 = 2,
  EltTyFloat32 = 7,
  EltTyBool16 = 10,
  EltTyInt8 = 1,
  EltTyBool32 = 11,
  EltTyBool64 = 12,
  EltTyInvalid = 0,
  EltTyBFloat16 = 13,
  EltTyMFloat8 = 14,
  EltTyInt32 = 3,
#endif

#ifdef LLVM_GET_SVE_MEMELTTYPES
  MemEltTyInt8 = 1,
  MemEltTyInt32 = 3,
  MemEltTyInt16 = 2,
  MemEltTyInt64 = 4,
  MemEltTyDefault = 0,
#endif

#ifdef LLVM_GET_SVE_MERGETYPES
  MergeAny = 1,
  MergeNone = 0,
  MergeOp1 = 2,
  MergeZero = 3,
  MergeZeroExp = 5,
  MergeAnyExp = 4,
#endif

