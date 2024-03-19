; ModuleID = '/Users/ttao/local_src/personal_upstream/llvm-project/clang/test/CodeGen/ms-intrinsics.c'
source_filename = "/Users/ttao/local_src/personal_upstream/llvm-project/clang/test/CodeGen/ms-intrinsics.c"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows-msvc"

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: write)
define dso_local void @test__stosb(ptr noundef writeonly %Dest, i8 noundef zeroext %Data, i32 noundef %Count) local_unnamed_addr #0 {
entry:
  tail call void @llvm.memset.p0.i32(ptr align 1 %Dest, i8 %Data, i32 %Count, i1 true)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg) #1

; Function Attrs: minsize nounwind optsize
define dso_local void @test__movsb(ptr noundef %Dest, ptr noundef %Src, i32 noundef %Count) local_unnamed_addr #2 {
entry:
  %0 = tail call { ptr, ptr, i32 } asm sideeffect "xchg $(%esi, $1$|$1, esi$)\0Arep movsb\0Axchg $(%esi, $1$|$1, esi$)", "={di},=r,={cx},{di},1,{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %Dest, ptr %Src, i32 %Count) #13, !srcloc !4
  ret void
}

; Function Attrs: minsize nounwind optsize
define dso_local void @test__stosw(ptr noundef %Dest, i16 noundef zeroext %Data, i32 noundef %Count) local_unnamed_addr #2 {
entry:
  %0 = tail call { ptr, i32 } asm sideeffect "rep stosw", "={di},={cx},{ax},{di},{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(i16 %Data, ptr %Dest, i32 %Count) #13, !srcloc !5
  ret void
}

; Function Attrs: minsize nounwind optsize
define dso_local void @test__movsw(ptr noundef %Dest, ptr noundef %Src, i32 noundef %Count) local_unnamed_addr #2 {
entry:
  %0 = tail call { ptr, ptr, i32 } asm sideeffect "xchg $(%esi, $1$|$1, esi$)\0Arep movsw\0Axchg $(%esi, $1$|$1, esi$)", "={di},=r,={cx},{di},1,{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %Dest, ptr %Src, i32 %Count) #13, !srcloc !6
  ret void
}

; Function Attrs: minsize nounwind optsize
define dso_local void @test__stosd(ptr noundef %Dest, i32 noundef %Data, i32 noundef %Count) local_unnamed_addr #2 {
entry:
  %0 = tail call { ptr, i32 } asm sideeffect "rep stos$(l$|d$)", "={di},={cx},{ax},{di},{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(i32 %Data, ptr %Dest, i32 %Count) #13, !srcloc !7
  ret void
}

; Function Attrs: minsize nounwind optsize
define dso_local void @test__movsd(ptr noundef %Dest, ptr noundef %Src, i32 noundef %Count) local_unnamed_addr #2 {
entry:
  %0 = tail call { ptr, ptr, i32 } asm sideeffect "xchg $(%esi, $1$|$1, esi$)\0Arep movs$(l$|d$)\0Axchg $(%esi, $1$|$1, esi$)", "={di},=r,={cx},{di},1,{cx},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %Dest, ptr %Src, i32 %Count) #13, !srcloc !8
  ret void
}

; Function Attrs: minsize noreturn nounwind optsize memory(inaccessiblemem: write)
define dso_local void @test__ud2() local_unnamed_addr #3 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #4

; Function Attrs: minsize noreturn nounwind optsize
define dso_local void @test__int2c() local_unnamed_addr #5 {
entry:
  tail call void asm sideeffect "int $$0x2c", ""() #14
  unreachable
}

; Function Attrs: minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(none)
define dso_local ptr @test_ReturnAddress() local_unnamed_addr #6 {
entry:
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.returnaddress(i32 immarg) #7

; Function Attrs: minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(none)
define dso_local ptr @test_AddressOfReturnAddress() local_unnamed_addr #6 {
entry:
  %0 = tail call ptr @llvm.addressofreturnaddress.p0()
  ret ptr %0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.addressofreturnaddress.p0() #7

; Function Attrs: minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(argmem: write)
define dso_local noundef zeroext i8 @test_BitScanForward(ptr nocapture noundef writeonly %Index, i32 noundef %Mask) local_unnamed_addr #8 {
entry:
  %0 = icmp eq i32 %Mask, 0
  br i1 %0, label %bitscan_end, label %bitscan_not_zero

bitscan_end:                                      ; preds = %bitscan_not_zero, %entry
  %bitscan_result = phi i8 [ 0, %entry ], [ 1, %bitscan_not_zero ]
  ret i8 %bitscan_result

bitscan_not_zero:                                 ; preds = %entry
  %incdec.ptr = getelementptr inbounds i8, ptr %Index, i32 4
  %1 = tail call i32 @llvm.cttz.i32(i32 %Mask, i1 true), !range !9
  store i32 %1, ptr %incdec.ptr, align 4
  br label %bitscan_end
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) #9

; Function Attrs: minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(argmem: write)
define dso_local noundef zeroext i8 @test_BitScanReverse(ptr nocapture noundef writeonly %Index, i32 noundef %Mask) local_unnamed_addr #8 {
entry:
  %0 = icmp eq i32 %Mask, 0
  br i1 %0, label %bitscan_end, label %bitscan_not_zero

bitscan_end:                                      ; preds = %bitscan_not_zero, %entry
  %bitscan_result = phi i8 [ 0, %entry ], [ 1, %bitscan_not_zero ]
  ret i8 %bitscan_result

bitscan_not_zero:                                 ; preds = %entry
  %incdec.ptr = getelementptr inbounds i8, ptr %Index, i32 4
  %1 = tail call i32 @llvm.ctlz.i32(i32 %Mask, i1 true), !range !9
  %2 = xor i32 %1, 31
  store i32 %2, ptr %incdec.ptr, align 4
  br label %bitscan_end
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #9

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local ptr @test_InterlockedExchangePointer(ptr nocapture noundef %Target, ptr noundef %Value) local_unnamed_addr #10 {
entry:
  %0 = ptrtoint ptr %Value to i32
  %1 = atomicrmw xchg ptr %Target, i32 %0 seq_cst, align 4
  %2 = inttoptr i32 %1 to ptr
  ret ptr %2
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local ptr @test_InterlockedCompareExchangePointer(ptr noundef %Destination, ptr noundef %Exchange, ptr noundef %Comparand) local_unnamed_addr #11 {
entry:
  %0 = ptrtoint ptr %Exchange to i32
  %1 = ptrtoint ptr %Comparand to i32
  %2 = cmpxchg volatile ptr %Destination, i32 %1, i32 %0 seq_cst seq_cst, align 4
  %3 = extractvalue { i32, i1 } %2, 0
  %4 = inttoptr i32 %3 to ptr
  ret ptr %4
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local ptr @test_InterlockedCompareExchangePointer_nf(ptr noundef %Destination, ptr noundef %Exchange, ptr noundef %Comparand) local_unnamed_addr #11 {
entry:
  %0 = ptrtoint ptr %Exchange to i32
  %1 = ptrtoint ptr %Comparand to i32
  %2 = cmpxchg volatile ptr %Destination, i32 %1, i32 %0 monotonic monotonic, align 4
  %3 = extractvalue { i32, i1 } %2, 0
  %4 = inttoptr i32 %3 to ptr
  ret ptr %4
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedExchange8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xchg ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedExchange16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xchg ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedExchange(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xchg ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedExchangeAdd8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw add ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedExchangeAdd16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw add ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedExchangeAdd(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw add ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedExchangeSub8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedExchangeSub16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedExchangeSub(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedOr8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw or ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedOr16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw or ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedOr(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw or ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedXor8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xor ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedXor16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xor ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedXor(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xor ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i8 @test_InterlockedAnd8(ptr nocapture noundef %value, i8 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw and ptr %value, i8 %mask seq_cst, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedAnd16(ptr nocapture noundef %value, i16 noundef signext %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw and ptr %value, i16 %mask seq_cst, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedAnd(ptr nocapture noundef %value, i32 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw and ptr %value, i32 %mask seq_cst, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local signext i8 @test_InterlockedCompareExchange8(ptr noundef %Destination, i8 noundef signext %Exchange, i8 noundef signext %Comperand) local_unnamed_addr #11 {
entry:
  %0 = cmpxchg volatile ptr %Destination, i8 %Comperand, i8 %Exchange seq_cst seq_cst, align 1
  %1 = extractvalue { i8, i1 } %0, 0
  ret i8 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local signext i16 @test_InterlockedCompareExchange16(ptr noundef %Destination, i16 noundef signext %Exchange, i16 noundef signext %Comperand) local_unnamed_addr #11 {
entry:
  %0 = cmpxchg volatile ptr %Destination, i16 %Comperand, i16 %Exchange seq_cst seq_cst, align 2
  %1 = extractvalue { i16, i1 } %0, 0
  ret i16 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local i32 @test_InterlockedCompareExchange(ptr noundef %Destination, i32 noundef %Exchange, i32 noundef %Comperand) local_unnamed_addr #11 {
entry:
  %0 = cmpxchg volatile ptr %Destination, i32 %Comperand, i32 %Exchange seq_cst seq_cst, align 4
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local i64 @test_InterlockedCompareExchange64(ptr noundef %Destination, i64 noundef %Exchange, i64 noundef %Comperand) local_unnamed_addr #11 {
entry:
  %0 = cmpxchg volatile ptr %Destination, i64 %Comperand, i64 %Exchange seq_cst seq_cst, align 8
  %1 = extractvalue { i64, i1 } %0, 0
  ret i64 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedIncrement16(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %incdec.ptr = getelementptr inbounds i8, ptr %Addend, i32 2
  %0 = atomicrmw add ptr %incdec.ptr, i16 1 seq_cst, align 2
  %1 = add i16 %0, 1
  ret i16 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedIncrement(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %incdec.ptr = getelementptr inbounds i8, ptr %Addend, i32 4
  %0 = atomicrmw add ptr %incdec.ptr, i32 1 seq_cst, align 4
  %1 = add i32 %0, 1
  ret i32 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local signext i16 @test_InterlockedDecrement16(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %Addend, i16 1 seq_cst, align 2
  %1 = add i16 %0, -1
  ret i16 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i32 @test_InterlockedDecrement(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %Addend, i32 1 seq_cst, align 4
  %1 = add i32 %0, -1
  ret i32 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local signext i8 @test_iso_volatile_load8(ptr noundef %p) local_unnamed_addr #11 {
entry:
  %0 = load volatile i8, ptr %p, align 1
  ret i8 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local signext i16 @test_iso_volatile_load16(ptr noundef %p) local_unnamed_addr #11 {
entry:
  %0 = load volatile i16, ptr %p, align 2
  ret i16 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local i32 @test_iso_volatile_load32(ptr noundef %p) local_unnamed_addr #11 {
entry:
  %0 = load volatile i32, ptr %p, align 4
  ret i32 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local i64 @test_iso_volatile_load64(ptr noundef %p) local_unnamed_addr #11 {
entry:
  %0 = load volatile i64, ptr %p, align 8
  ret i64 %0
}

; Function Attrs: minsize nofree norecurse nounwind optsize memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @test_iso_volatile_store8(ptr noundef %p, i8 noundef signext %v) local_unnamed_addr #12 {
entry:
  store volatile i8 %v, ptr %p, align 1
  ret void
}

; Function Attrs: minsize nofree norecurse nounwind optsize memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @test_iso_volatile_store16(ptr noundef %p, i16 noundef signext %v) local_unnamed_addr #12 {
entry:
  store volatile i16 %v, ptr %p, align 2
  ret void
}

; Function Attrs: minsize nofree norecurse nounwind optsize memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @test_iso_volatile_store32(ptr noundef %p, i32 noundef %v) local_unnamed_addr #12 {
entry:
  store volatile i32 %v, ptr %p, align 4
  ret void
}

; Function Attrs: minsize nofree norecurse nounwind optsize memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @test_iso_volatile_store64(ptr noundef %p, i64 noundef %v) local_unnamed_addr #12 {
entry:
  store volatile i64 %v, ptr %p, align 8
  ret void
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedExchange64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xchg ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedExchangeAdd64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw add ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedExchangeSub64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedOr64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw or ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedXor64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw xor ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedAnd64(ptr nocapture noundef %value, i64 noundef %mask) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw and ptr %value, i64 %mask seq_cst, align 8
  ret i64 %0
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedIncrement64(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw add ptr %Addend, i64 1 seq_cst, align 8
  %1 = add i64 %0, 1
  ret i64 %1
}

; Function Attrs: minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite)
define dso_local i64 @test_InterlockedDecrement64(ptr nocapture noundef %Addend) local_unnamed_addr #10 {
entry:
  %0 = atomicrmw sub ptr %Addend, i64 1 seq_cst, align 8
  %1 = add i64 %0, -1
  ret i64 %1
}

; Function Attrs: minsize nounwind optsize
define dso_local i32 @test_InterlockedExchange_HLEAcquire(ptr noundef %Target, i32 noundef %Value) local_unnamed_addr #2 {
entry:
  %0 = tail call i32 asm sideeffect ".byte 0xf2 ; lock ; xchg $($0, $1$|$1, $0$)", "=r,=*m,0,*m,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %Target, i32 %Value, ptr elementtype(i32) %Target) #13, !srcloc !10
  ret i32 %0
}

; Function Attrs: minsize nounwind optsize
define dso_local i32 @test_InterlockedExchange_HLERelease(ptr noundef %Target, i32 noundef %Value) local_unnamed_addr #2 {
entry:
  %0 = tail call i32 asm sideeffect ".byte 0xf3 ; lock ; xchg $($0, $1$|$1, $0$)", "=r,=*m,0,*m,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %Target, i32 %Value, ptr elementtype(i32) %Target) #13, !srcloc !11
  ret i32 %0
}

; Function Attrs: minsize nounwind optsize
define dso_local i32 @test_InterlockedCompareExchange_HLEAcquire(ptr noundef %Destination, i32 noundef %Exchange, i32 noundef %Comparand) local_unnamed_addr #2 {
entry:
  %0 = tail call i32 asm sideeffect ".byte 0xf2 ; lock ; cmpxchg $($2, $1$|$1, $2$)", "={ax},=*m,r,{ax},*m,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %Destination, i32 %Exchange, i32 %Comparand, ptr elementtype(i32) %Destination) #13, !srcloc !12
  ret i32 %0
}

; Function Attrs: minsize nounwind optsize
define dso_local i32 @test_InterlockedCompareExchange_HLERelease(ptr noundef %Destination, i32 noundef %Exchange, i32 noundef %Comparand) local_unnamed_addr #2 {
entry:
  %0 = tail call i32 asm sideeffect ".byte 0xf3 ; lock ; cmpxchg $($2, $1$|$1, $2$)", "={ax},=*m,r,{ax},*m,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %Destination, i32 %Exchange, i32 %Comparand, ptr elementtype(i32) %Destination) #13, !srcloc !13
  ret i32 %0
}

; Function Attrs: minsize noreturn nounwind optsize
define dso_local void @test__fastfail() local_unnamed_addr #5 {
entry:
  tail call void asm sideeffect "int $$0x29", "{cx}"(i32 42) #14
  unreachable
}

attributes #0 = { minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: write) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { minsize nounwind optsize "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #3 = { minsize noreturn nounwind optsize memory(inaccessiblemem: write) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #4 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #5 = { minsize noreturn nounwind optsize "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #6 = { minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(none) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #7 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { minsize mustprogress nofree norecurse nosync nounwind optsize willreturn memory(argmem: write) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #9 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #11 = { minsize mustprogress nofree norecurse nounwind optsize willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #12 = { minsize nofree norecurse nounwind optsize memory(argmem: readwrite, inaccessiblemem: readwrite) "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+x87" }
attributes #13 = { nounwind }
attributes #14 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, !"MaxTLSAlign", i32 65536}
!3 = !{!"clang version 19.0.0git (git@github.com:tltao/llvm-project.git ea72c082bc29fdceca33f37477b7588f31630a5f)"}
!4 = !{i64 109437, i64 109490, i64 109527}
!5 = !{i64 111630}
!6 = !{i64 110789, i64 110842, i64 110879}
!7 = !{i64 111264}
!8 = !{i64 110112, i64 110165, i64 110206}
!9 = !{i32 0, i32 33}
!10 = !{i64 168252}
!11 = !{i64 168520}
!12 = !{i64 169689}
!13 = !{i64 170075}
