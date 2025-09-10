; ModuleID = 'MultiSource/Benchmarks/Prolangs-C/agrep/asearch.c'
source_filename = "MultiSource/Benchmarks/Prolangs-C/agrep/asearch.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@D_endpos = external local_unnamed_addr global i32, align 4
@Init1 = external local_unnamed_addr global i32, align 4
@NO_ERR_MASK = external local_unnamed_addr global i32, align 4
@Init = external local_unnamed_addr global [0 x i32], align 4
@Mask = external local_unnamed_addr global [0 x i32], align 4
@AND = external local_unnamed_addr global i32, align 4
@endposition = external local_unnamed_addr global i32, align 4
@INVERSE = external local_unnamed_addr global i32, align 4
@FILENAMEONLY = external local_unnamed_addr global i32, align 4
@num_of_matched = external local_unnamed_addr global i32, align 4
@CurrentFileName = external global [0 x i8], align 1
@TRUNCATE = external local_unnamed_addr global i32, align 4
@I = external local_unnamed_addr global i32, align 4
@DELIMITER = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define dso_local void @asearch0(ptr noundef readonly captures(none) %old_D_pat, i32 noundef %text, i32 noundef %D) local_unnamed_addr #0 {
entry:
  %A = alloca [10 x i32], align 4
  %B = alloca [10 x i32], align 4
  %buffer = alloca [98305 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %A) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %B) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %buffer) #6
  %call = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %old_D_pat) #7
  %conv = trunc i64 %call to i32
  %arrayidx = getelementptr inbounds nuw i8, ptr %buffer, i64 49151
  store i8 10, ptr %arrayidx, align 1, !tbaa !6
  %0 = load i32, ptr @D_endpos, align 4, !tbaa !9
  %cmp461 = icmp ugt i32 %conv, 1
  br i1 %cmp461, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.0463 = phi i32 [ %inc, %for.body ], [ 1, %entry ]
  %D_Mask.0462 = phi i32 [ %or, %for.body ], [ %0, %entry ]
  %shl = shl i32 %D_Mask.0462, 1
  %or = or i32 %shl, %D_Mask.0462
  %inc = add nuw i32 %i.0463, 1
  %exitcond.not = icmp eq i32 %inc, %conv
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !11

for.end:                                          ; preds = %for.body, %entry
  %D_Mask.0.lcssa = phi i32 [ %0, %entry ], [ %or, %for.body ]
  %1 = load i32, ptr @Init1, align 4, !tbaa !9
  %2 = load i32, ptr @NO_ERR_MASK, align 4, !tbaa !9
  %3 = load i32, ptr @Init, align 4, !tbaa !9
  %4 = add i32 %D, 1
  %wide.trip.count = zext i32 %4 to i64
  %min.iters.check = icmp ult i32 %4, 8
  br i1 %min.iters.check, label %for.body5.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.end
  %n.vec = and i64 %wide.trip.count, 4294967288
  %broadcast.splatinsert = insertelement <4 x i32> poison, i32 %3, i64 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %5 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %index
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store <4 x i32> %broadcast.splat, ptr %5, align 4, !tbaa !9
  store <4 x i32> %broadcast.splat, ptr %6, align 4, !tbaa !9
  %7 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %index
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store <4 x i32> %broadcast.splat, ptr %7, align 4, !tbaa !9
  store <4 x i32> %broadcast.splat, ptr %8, align 4, !tbaa !9
  %index.next = add nuw i64 %index, 8
  %9 = icmp eq i64 %index.next, %n.vec
  br i1 %9, label %middle.block, label %vector.body, !llvm.loop !13

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %while.cond.preheader, label %for.body5.preheader

for.body5.preheader:                              ; preds = %for.end, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %for.end ], [ %n.vec, %middle.block ]
  br label %for.body5

while.cond.preheader:                             ; preds = %for.body5, %middle.block
  %not = xor i32 %D_Mask.0.lcssa, -1
  %add.ptr = getelementptr inbounds nuw i8, ptr %buffer, i64 49152
  %call12481 = call i32 @fill_buf(i32 noundef %text, ptr noundef nonnull %add.ptr, i32 noundef 49152) #6
  %cmp13482 = icmp sgt i32 %call12481, 0
  br i1 %cmp13482, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  %sext = shl i64 %call, 32
  %conv20 = ashr exact i64 %sext, 32
  %cmp43.not465 = icmp eq i32 %D, 0
  %idxprom77 = zext i32 %D to i64
  %arrayidx78 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %idxprom77
  %arrayidx205 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %idxprom77
  %10 = xor i32 %conv, -1
  %min.iters.check542 = icmp ult i32 %4, 8
  %n.vec545 = and i64 %wide.trip.count, 4294967288
  %cmp.n552 = icmp eq i64 %n.vec545, %wide.trip.count
  %min.iters.check529 = icmp ult i32 %4, 8
  %n.vec532 = and i64 %wide.trip.count, 4294967288
  %cmp.n539 = icmp eq i64 %n.vec532, %wide.trip.count
  br label %while.body

for.body5:                                        ; preds = %for.body5.preheader, %for.body5
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body5 ], [ %indvars.iv.ph, %for.body5.preheader ]
  %arrayidx6 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv
  store i32 %3, ptr %arrayidx6, align 4, !tbaa !9
  %arrayidx8 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv
  store i32 %3, ptr %arrayidx8, align 4, !tbaa !9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond488.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond488.not, label %while.cond.preheader, label %for.body5, !llvm.loop !16

while.body:                                       ; preds = %while.body.lr.ph, %if.end311
  %call12486 = phi i32 [ %call12481, %while.body.lr.ph ], [ %call12, %if.end311 ]
  %j.0485 = phi i32 [ 0, %while.body.lr.ph ], [ %j.1.lcssa, %if.end311 ]
  %lasti.0484 = phi i32 [ 49152, %while.body.lr.ph ], [ %lasti.4, %if.end311 ]
  %tobool.not483 = phi i32 [ 49151, %while.body.lr.ph ], [ 49152, %if.end311 ]
  %add = add nuw nsw i32 %call12486, 49152
  %cmp15 = icmp samesign ult i32 %call12486, 49152
  br i1 %cmp15, label %if.then17, label %if.end26

if.then17:                                        ; preds = %while.body
  %idx.ext = zext nneg i32 %add to i64
  %add.ptr19 = getelementptr inbounds nuw i8, ptr %buffer, i64 %idx.ext
  %call21 = call ptr @strncpy(ptr noundef nonnull %add.ptr19, ptr noundef nonnull %old_D_pat, i64 noundef %conv20) #6
  %add22 = add i32 %add, %conv
  %idxprom23 = zext i32 %add22 to i64
  %arrayidx24 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom23
  store i8 0, ptr %arrayidx24, align 1, !tbaa !6
  br label %if.end26

if.end26:                                         ; preds = %if.then17, %while.body
  %end.0 = phi i32 [ %add22, %if.then17 ], [ %add, %while.body ]
  %cmp28475 = icmp ult i32 %tobool.not483, %end.0
  br i1 %cmp28475, label %while.body30.lr.ph, label %while.end

while.body30.lr.ph:                               ; preds = %if.end26
  %sub98 = add nuw nsw i32 %call12486, 49151
  %.pre = load i32, ptr %B, align 4, !tbaa !9
  br label %while.body30

while.body30:                                     ; preds = %while.body30.lr.ph, %if.end287
  %11 = phi i32 [ %.pre, %while.body30.lr.ph ], [ %58, %if.end287 ]
  %j.1478 = phi i32 [ %j.0485, %while.body30.lr.ph ], [ %j.3, %if.end287 ]
  %i.2477 = phi i32 [ %tobool.not483, %while.body30.lr.ph ], [ %inc155, %if.end287 ]
  %lasti.1476 = phi i32 [ %lasti.0484, %while.body30.lr.ph ], [ %lasti.3, %if.end287 ]
  %inc31 = add nuw i32 %i.2477, 1
  %idxprom32 = zext i32 %i.2477 to i64
  %arrayidx33 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom32
  %12 = load i8, ptr %arrayidx33, align 1, !tbaa !6
  %idxprom35 = zext i8 %12 to i64
  %arrayidx36 = getelementptr inbounds nuw [0 x i32], ptr @Mask, i64 0, i64 %idxprom35
  %13 = load i32, ptr %arrayidx36, align 4, !tbaa !9
  %and = and i32 %11, %1
  %shr = lshr i32 %11, 1
  %and39 = and i32 %shr, %13
  %or40 = or i32 %and39, %and
  store i32 %or40, ptr %A, align 4, !tbaa !9
  br i1 %cmp43.not465, label %for.end71, label %for.body45

for.body45:                                       ; preds = %while.body30, %for.body45
  %14 = phi i32 [ %or66, %for.body45 ], [ %or40, %while.body30 ]
  %15 = phi i32 [ %16, %for.body45 ], [ %11, %while.body30 ]
  %indvars.iv489 = phi i64 [ %indvars.iv.next490, %for.body45 ], [ 1, %while.body30 ]
  %arrayidx47 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv489
  %16 = load i32, ptr %arrayidx47, align 4, !tbaa !9
  %and48 = and i32 %16, %1
  %or57 = or i32 %14, %15
  %shr58 = lshr i32 %or57, 1
  %and59 = and i32 %shr58, %2
  %shr63 = lshr i32 %16, 1
  %and64 = and i32 %shr63, %13
  %17 = or i32 %and48, %and64
  %18 = or i32 %17, %and59
  %or66 = or i32 %18, %15
  %arrayidx68 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv489
  store i32 %or66, ptr %arrayidx68, align 4, !tbaa !9
  %indvars.iv.next490 = add nuw nsw i64 %indvars.iv489, 1
  %exitcond494.not = icmp eq i64 %indvars.iv.next490, %wide.trip.count
  br i1 %exitcond494.not, label %for.end71, label %for.body45, !llvm.loop !17

for.end71:                                        ; preds = %for.body45, %while.body30
  %and73 = and i32 %or40, %0
  %tobool74.not = icmp eq i32 %and73, 0
  br i1 %tobool74.not, label %if.end154, label %if.then75

if.then75:                                        ; preds = %for.end71
  %inc76 = add nsw i32 %j.1478, 1
  %19 = load i32, ptr %arrayidx78, align 4, !tbaa !9
  %20 = load i32, ptr @AND, align 4, !tbaa !9
  %cmp79 = icmp eq i32 %20, 1
  %.pre523 = load i32, ptr @endposition, align 4
  %and81 = and i32 %.pre523, %19
  %cmp82 = icmp eq i32 %and81, %.pre523
  %or.cond = select i1 %cmp79, i1 %cmp82, i1 false
  br i1 %or.cond, label %if.then89, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.then75
  %cmp84 = icmp eq i32 %20, 0
  %tobool87 = icmp ne i32 %and81, 0
  %21 = select i1 %cmp84, i1 %tobool87, i1 false
  %land.ext = zext i1 %21 to i32
  %22 = load i32, ptr @INVERSE, align 4, !tbaa !9
  %tobool88.not = icmp eq i32 %22, %land.ext
  br i1 %tobool88.not, label %if.end104, label %if.then89

if.then89:                                        ; preds = %if.then75, %lor.lhs.false
  %23 = load i32, ptr @FILENAMEONLY, align 4, !tbaa !9
  %tobool90.not = icmp eq i32 %23, 0
  br i1 %tobool90.not, label %if.end94, label %cleanup.sink.split

if.end94:                                         ; preds = %if.then89
  %cmp99.not = icmp slt i32 %lasti.1476, %sub98
  br i1 %cmp99.not, label %if.then101, label %if.end104

if.then101:                                       ; preds = %if.end94
  %sub96 = sub i32 %i.2477, %conv
  call void @output(ptr noundef nonnull %buffer, i32 noundef %lasti.1476, i32 noundef %sub96, i32 noundef %inc76) #6
  br label %if.end104

if.end104:                                        ; preds = %if.end94, %if.then101, %lor.lhs.false
  %24 = load i32, ptr @Init, align 4, !tbaa !9
  br i1 %min.iters.check542, label %for.body109.preheader, label %vector.ph543

vector.ph543:                                     ; preds = %if.end104
  %broadcast.splatinsert546 = insertelement <4 x i32> poison, i32 %24, i64 0
  %broadcast.splat547 = shufflevector <4 x i32> %broadcast.splatinsert546, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %vector.body548

vector.body548:                                   ; preds = %vector.body548, %vector.ph543
  %index549 = phi i64 [ 0, %vector.ph543 ], [ %index.next550, %vector.body548 ]
  %25 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %index549
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  store <4 x i32> %broadcast.splat547, ptr %25, align 4, !tbaa !9
  store <4 x i32> %broadcast.splat547, ptr %26, align 4, !tbaa !9
  %index.next550 = add nuw i64 %index549, 8
  %27 = icmp eq i64 %index.next550, %n.vec545
  br i1 %27, label %middle.block551, label %vector.body548, !llvm.loop !18

middle.block551:                                  ; preds = %vector.body548
  br i1 %cmp.n552, label %for.end114, label %for.body109.preheader

for.body109.preheader:                            ; preds = %if.end104, %middle.block551
  %indvars.iv495.ph = phi i64 [ 0, %if.end104 ], [ %n.vec545, %middle.block551 ]
  br label %for.body109

for.body109:                                      ; preds = %for.body109.preheader, %for.body109
  %indvars.iv495 = phi i64 [ %indvars.iv.next496, %for.body109 ], [ %indvars.iv495.ph, %for.body109.preheader ]
  %arrayidx111 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv495
  store i32 %24, ptr %arrayidx111, align 4, !tbaa !9
  %indvars.iv.next496 = add nuw nsw i64 %indvars.iv495, 1
  %exitcond499.not = icmp eq i64 %indvars.iv.next496, %wide.trip.count
  br i1 %exitcond499.not, label %for.end114, label %for.body109, !llvm.loop !19

for.end114:                                       ; preds = %for.body109, %middle.block551
  %sub105 = sub i32 %inc31, %conv
  %28 = load i32, ptr %B, align 4, !tbaa !9
  %and116 = and i32 %28, %1
  %shr118 = lshr i32 %28, 1
  %and119 = and i32 %shr118, %13
  %or120 = or i32 %and119, %and116
  %and121 = and i32 %or120, %not
  store i32 %and121, ptr %A, align 4, !tbaa !9
  br i1 %cmp43.not465, label %if.end154, label %for.body126.lr.ph

for.body126.lr.ph:                                ; preds = %for.end114
  %29 = load i32, ptr @Init1, align 4, !tbaa !9
  br label %for.body126

for.body126:                                      ; preds = %for.body126.lr.ph, %for.body126
  %30 = phi i32 [ %and121, %for.body126.lr.ph ], [ %or148, %for.body126 ]
  %31 = phi i32 [ %28, %for.body126.lr.ph ], [ %32, %for.body126 ]
  %indvars.iv500 = phi i64 [ 1, %for.body126.lr.ph ], [ %indvars.iv.next501, %for.body126 ]
  %arrayidx128 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv500
  %32 = load i32, ptr %arrayidx128, align 4, !tbaa !9
  %and129 = and i32 %32, %29
  %or139 = or i32 %30, %31
  %shr140 = lshr i32 %or139, 1
  %and141 = and i32 %shr140, %2
  %shr145 = lshr i32 %32, 1
  %and146 = and i32 %shr145, %13
  %33 = or i32 %and129, %and146
  %34 = or i32 %33, %and141
  %or148 = or i32 %34, %31
  %arrayidx150 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv500
  store i32 %or148, ptr %arrayidx150, align 4, !tbaa !9
  %indvars.iv.next501 = add nuw nsw i64 %indvars.iv500, 1
  %exitcond505.not = icmp eq i64 %indvars.iv.next501, %wide.trip.count
  br i1 %exitcond505.not, label %if.end154, label %for.body126, !llvm.loop !20

if.end154:                                        ; preds = %for.body126, %for.end114, %for.end71
  %35 = phi i32 [ %or40, %for.end71 ], [ %and121, %for.end114 ], [ %and121, %for.body126 ]
  %lasti.2 = phi i32 [ %lasti.1476, %for.end71 ], [ %sub105, %for.end114 ], [ %sub105, %for.body126 ]
  %j.2 = phi i32 [ %j.1478, %for.end71 ], [ %inc76, %for.end114 ], [ %inc76, %for.body126 ]
  %inc155 = add i32 %i.2477, 2
  %idxprom156 = zext i32 %inc31 to i64
  %arrayidx157 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom156
  %36 = load i8, ptr %arrayidx157, align 1, !tbaa !6
  %idxprom159 = zext i8 %36 to i64
  %arrayidx160 = getelementptr inbounds nuw [0 x i32], ptr @Mask, i64 0, i64 %idxprom159
  %37 = load i32, ptr %arrayidx160, align 4, !tbaa !9
  %and162 = and i32 %35, %1
  %shr164 = lshr i32 %35, 1
  %and165 = and i32 %shr164, %37
  %or166 = or i32 %and165, %and162
  store i32 %or166, ptr %B, align 4, !tbaa !9
  br i1 %cmp43.not465, label %for.end198, label %for.body171

for.body171:                                      ; preds = %if.end154, %for.body171
  %38 = phi i32 [ %or193, %for.body171 ], [ %or166, %if.end154 ]
  %39 = phi i32 [ %40, %for.body171 ], [ %35, %if.end154 ]
  %indvars.iv506 = phi i64 [ %indvars.iv.next507, %for.body171 ], [ 1, %if.end154 ]
  %arrayidx173 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv506
  %40 = load i32, ptr %arrayidx173, align 4, !tbaa !9
  %and174 = and i32 %40, %1
  %or184 = or i32 %38, %39
  %shr185 = lshr i32 %or184, 1
  %and186 = and i32 %shr185, %2
  %shr190 = lshr i32 %40, 1
  %and191 = and i32 %shr190, %37
  %41 = or i32 %and174, %and191
  %42 = or i32 %41, %and186
  %or193 = or i32 %42, %39
  %arrayidx195 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv506
  store i32 %or193, ptr %arrayidx195, align 4, !tbaa !9
  %indvars.iv.next507 = add nuw nsw i64 %indvars.iv506, 1
  %exitcond511.not = icmp eq i64 %indvars.iv.next507, %wide.trip.count
  br i1 %exitcond511.not, label %for.end198, label %for.body171, !llvm.loop !21

for.end198:                                       ; preds = %for.body171, %if.end154
  %and200 = and i32 %or166, %0
  %tobool201.not = icmp eq i32 %and200, 0
  br i1 %tobool201.not, label %if.end287, label %if.then202

if.then202:                                       ; preds = %for.end198
  %inc203 = add nsw i32 %j.2, 1
  %43 = load i32, ptr %arrayidx205, align 4, !tbaa !9
  %44 = load i32, ptr @AND, align 4, !tbaa !9
  %cmp206 = icmp eq i32 %44, 1
  %.pre524 = load i32, ptr @endposition, align 4
  %and209 = and i32 %.pre524, %43
  %cmp210 = icmp eq i32 %and209, %.pre524
  %or.cond527 = select i1 %cmp206, i1 %cmp210, i1 false
  br i1 %or.cond527, label %if.then222, label %lor.lhs.false212

lor.lhs.false212:                                 ; preds = %if.then202
  %cmp213 = icmp eq i32 %44, 0
  %tobool217 = icmp ne i32 %and209, 0
  %45 = select i1 %cmp213, i1 %tobool217, i1 false
  %land.ext219 = zext i1 %45 to i32
  %46 = load i32, ptr @INVERSE, align 4, !tbaa !9
  %tobool221.not = icmp eq i32 %46, %land.ext219
  br i1 %tobool221.not, label %if.end237, label %if.then222

if.then222:                                       ; preds = %if.then202, %lor.lhs.false212
  %47 = load i32, ptr @FILENAMEONLY, align 4, !tbaa !9
  %tobool223.not = icmp eq i32 %47, 0
  br i1 %tobool223.not, label %if.end227, label %cleanup.sink.split

if.end227:                                        ; preds = %if.then222
  %cmp232.not = icmp slt i32 %lasti.2, %sub98
  br i1 %cmp232.not, label %if.then234, label %if.end237

if.then234:                                       ; preds = %if.end227
  %sub229 = add i32 %inc155, %10
  call void @output(ptr noundef nonnull %buffer, i32 noundef %lasti.2, i32 noundef %sub229, i32 noundef %inc203) #6
  br label %if.end237

if.end237:                                        ; preds = %if.end227, %if.then234, %lor.lhs.false212
  %48 = load i32, ptr @Init, align 4, !tbaa !9
  br i1 %min.iters.check529, label %for.body242.preheader, label %vector.ph530

vector.ph530:                                     ; preds = %if.end237
  %broadcast.splatinsert533 = insertelement <4 x i32> poison, i32 %48, i64 0
  %broadcast.splat534 = shufflevector <4 x i32> %broadcast.splatinsert533, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %vector.body535

vector.body535:                                   ; preds = %vector.body535, %vector.ph530
  %index536 = phi i64 [ 0, %vector.ph530 ], [ %index.next537, %vector.body535 ]
  %49 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %index536
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 16
  store <4 x i32> %broadcast.splat534, ptr %49, align 4, !tbaa !9
  store <4 x i32> %broadcast.splat534, ptr %50, align 4, !tbaa !9
  %index.next537 = add nuw i64 %index536, 8
  %51 = icmp eq i64 %index.next537, %n.vec532
  br i1 %51, label %middle.block538, label %vector.body535, !llvm.loop !22

middle.block538:                                  ; preds = %vector.body535
  br i1 %cmp.n539, label %for.end247, label %for.body242.preheader

for.body242.preheader:                            ; preds = %if.end237, %middle.block538
  %indvars.iv512.ph = phi i64 [ 0, %if.end237 ], [ %n.vec532, %middle.block538 ]
  br label %for.body242

for.body242:                                      ; preds = %for.body242.preheader, %for.body242
  %indvars.iv512 = phi i64 [ %indvars.iv.next513, %for.body242 ], [ %indvars.iv512.ph, %for.body242.preheader ]
  %arrayidx244 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv512
  store i32 %48, ptr %arrayidx244, align 4, !tbaa !9
  %indvars.iv.next513 = add nuw nsw i64 %indvars.iv512, 1
  %exitcond516.not = icmp eq i64 %indvars.iv.next513, %wide.trip.count
  br i1 %exitcond516.not, label %for.end247, label %for.body242, !llvm.loop !23

for.end247:                                       ; preds = %for.body242, %middle.block538
  %sub238 = sub i32 %inc155, %conv
  %52 = load i32, ptr %A, align 4, !tbaa !9
  %and249 = and i32 %52, %1
  %shr251 = lshr i32 %52, 1
  %and252 = and i32 %shr251, %37
  %or253 = or i32 %and252, %and249
  %and254 = and i32 %or253, %not
  store i32 %and254, ptr %B, align 4, !tbaa !9
  br i1 %cmp43.not465, label %if.end287, label %for.body259

for.body259:                                      ; preds = %for.end247, %for.body259
  %53 = phi i32 [ %or281, %for.body259 ], [ %and254, %for.end247 ]
  %54 = phi i32 [ %55, %for.body259 ], [ %52, %for.end247 ]
  %indvars.iv517 = phi i64 [ %indvars.iv.next518, %for.body259 ], [ 1, %for.end247 ]
  %arrayidx261 = getelementptr inbounds nuw [10 x i32], ptr %A, i64 0, i64 %indvars.iv517
  %55 = load i32, ptr %arrayidx261, align 4, !tbaa !9
  %and262 = and i32 %55, %1
  %or272 = or i32 %53, %54
  %shr273 = lshr i32 %or272, 1
  %and274 = and i32 %shr273, %2
  %shr278 = lshr i32 %55, 1
  %and279 = and i32 %shr278, %37
  %56 = or i32 %and262, %and279
  %57 = or i32 %56, %and274
  %or281 = or i32 %57, %54
  %arrayidx283 = getelementptr inbounds nuw [10 x i32], ptr %B, i64 0, i64 %indvars.iv517
  store i32 %or281, ptr %arrayidx283, align 4, !tbaa !9
  %indvars.iv.next518 = add nuw nsw i64 %indvars.iv517, 1
  %exitcond522.not = icmp eq i64 %indvars.iv.next518, %wide.trip.count
  br i1 %exitcond522.not, label %if.end287, label %for.body259, !llvm.loop !24

if.end287:                                        ; preds = %for.body259, %for.end247, %for.end198
  %58 = phi i32 [ %or166, %for.end198 ], [ %and254, %for.end247 ], [ %and254, %for.body259 ]
  %lasti.3 = phi i32 [ %lasti.2, %for.end198 ], [ %sub238, %for.end247 ], [ %sub238, %for.body259 ]
  %j.3 = phi i32 [ %j.2, %for.end198 ], [ %inc203, %for.end247 ], [ %inc203, %for.body259 ]
  %cmp28 = icmp ult i32 %inc155, %end.0
  br i1 %cmp28, label %while.body30, label %while.end, !llvm.loop !25

while.end:                                        ; preds = %if.end287, %if.end26
  %lasti.1.lcssa = phi i32 [ %lasti.0484, %if.end26 ], [ %lasti.3, %if.end287 ]
  %j.1.lcssa = phi i32 [ %j.0485, %if.end26 ], [ %j.3, %if.end287 ]
  br i1 %cmp15, label %if.end311, label %if.else

if.else:                                          ; preds = %while.end
  %sub292 = sub nsw i32 %add, %lasti.1.lcssa
  %cmp293 = icmp sgt i32 %sub292, 49152
  br i1 %cmp293, label %if.then295, label %if.end296

if.then295:                                       ; preds = %if.else
  store i32 1, ptr @TRUNCATE, align 4, !tbaa !9
  br label %if.end296

if.end296:                                        ; preds = %if.then295, %if.else
  %ResidueSize.0 = phi i32 [ 49152, %if.then295 ], [ %sub292, %if.else ]
  %ResidueSize.0.fr = freeze i32 %ResidueSize.0
  %idx.ext299 = sext i32 %ResidueSize.0.fr to i64
  %idx.neg = sub nsw i64 0, %idx.ext299
  %add.ptr300 = getelementptr inbounds i8, ptr %add.ptr, i64 %idx.neg
  %idx.ext302 = sext i32 %lasti.1.lcssa to i64
  %add.ptr303 = getelementptr inbounds i8, ptr %buffer, i64 %idx.ext302
  %call305 = call ptr @strncpy(ptr noundef nonnull %add.ptr300, ptr noundef nonnull %add.ptr303, i64 noundef %idx.ext299) #6
  %sub306 = sub nsw i32 49152, %ResidueSize.0.fr
  %cmp307 = icmp eq i32 %ResidueSize.0.fr, 49152
  %spec.select456 = select i1 %cmp307, i32 1, i32 %sub306
  br label %if.end311

if.end311:                                        ; preds = %if.end296, %while.end
  %lasti.4 = phi i32 [ 49152, %while.end ], [ %spec.select456, %if.end296 ]
  %call12 = call i32 @fill_buf(i32 noundef %text, ptr noundef nonnull %add.ptr, i32 noundef 49152) #6
  %cmp13 = icmp sgt i32 %call12, 0
  br i1 %cmp13, label %while.body, label %cleanup, !llvm.loop !26

cleanup.sink.split:                               ; preds = %if.then222, %if.then89
  %59 = load i32, ptr @num_of_matched, align 4, !tbaa !9
  %inc225 = add nsw i32 %59, 1
  store i32 %inc225, ptr @num_of_matched, align 4, !tbaa !9
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @CurrentFileName)
  br label %cleanup

cleanup:                                          ; preds = %if.end311, %cleanup.sink.split, %while.cond.preheader
  call void @llvm.lifetime.end.p0(ptr nonnull %buffer) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %B) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %A) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #2

declare i32 @fill_buf(i32 noundef, ptr noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strncpy(ptr noalias noundef returned writeonly, ptr noalias noundef readonly captures(none), i64 noundef) local_unnamed_addr #4

declare void @output(ptr noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local void @asearch(ptr noundef readonly captures(none) %old_D_pat, i32 noundef %text, i32 noundef %D) local_unnamed_addr #0 {
entry:
  %A = alloca [9 x i32], align 4
  %B = alloca [9 x i32], align 4
  %buffer = alloca [98305 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %A) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %B) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %buffer) #6
  %0 = load i32, ptr @I, align 4, !tbaa !9
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 -1, ptr @Init1, align 4, !tbaa !9
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %cmp1 = icmp ugt i32 %D, 4
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  tail call void @asearch0(ptr noundef %old_D_pat, i32 noundef %text, i32 noundef %D)
  br label %cleanup

if.end3:                                          ; preds = %if.end
  %call = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %old_D_pat) #7
  %conv = trunc i64 %call to i32
  %arrayidx = getelementptr inbounds nuw i8, ptr %buffer, i64 49151
  store i8 10, ptr %arrayidx, align 1, !tbaa !6
  %1 = load i32, ptr @D_endpos, align 4, !tbaa !9
  %cmp4684 = icmp ugt i32 %conv, 1
  br i1 %cmp4684, label %for.body, label %for.end

for.body:                                         ; preds = %if.end3, %for.body
  %i.0686 = phi i32 [ %inc, %for.body ], [ 1, %if.end3 ]
  %D_Mask.0685 = phi i32 [ %or, %for.body ], [ %1, %if.end3 ]
  %shl = shl i32 %D_Mask.0685, 1
  %or = or i32 %shl, %D_Mask.0685
  %inc = add nuw i32 %i.0686, 1
  %exitcond.not = icmp eq i32 %inc, %conv
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !27

for.end:                                          ; preds = %for.body, %if.end3
  %D_Mask.0.lcssa = phi i32 [ %1, %if.end3 ], [ %or, %for.body ]
  %2 = load i32, ptr @Init1, align 4, !tbaa !9
  %3 = load i32, ptr @NO_ERR_MASK, align 4, !tbaa !9
  %4 = load i32, ptr @Init, align 4, !tbaa !9
  %5 = add nuw nsw i32 %D, 1
  %wide.trip.count = zext nneg i32 %5 to i64
  br label %for.body9

while.cond.preheader:                             ; preds = %for.body9
  %not = xor i32 %D_Mask.0.lcssa, -1
  %add.ptr = getelementptr inbounds nuw i8, ptr %buffer, i64 49152
  %call16721 = call i32 @fill_buf(i32 noundef %text, ptr noundef nonnull %add.ptr, i32 noundef 49152) #6
  %cmp17722 = icmp sgt i32 %call16721, 0
  br i1 %cmp17722, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  %cmp23688 = icmp slt i32 %conv, 1
  %sext = shl i64 %call, 32
  %conv51 = ashr exact i64 %sext, 32
  %cmp78 = icmp eq i32 %D, 1
  %cmp91 = icmp eq i32 %D, 2
  %cmp104 = icmp eq i32 %D, 3
  %cmp138 = icmp eq i32 %D, 4
  %cmp186.not694 = icmp eq i32 %D, 0
  %arrayidx218 = getelementptr inbounds nuw i8, ptr %A, i64 4
  %arrayidx220 = getelementptr inbounds nuw i8, ptr %A, i64 8
  %arrayidx221 = getelementptr inbounds nuw i8, ptr %B, i64 8
  %arrayidx222 = getelementptr inbounds nuw i8, ptr %A, i64 12
  %arrayidx223 = getelementptr inbounds nuw i8, ptr %B, i64 12
  %arrayidx224 = getelementptr inbounds nuw i8, ptr %A, i64 16
  %arrayidx225 = getelementptr inbounds nuw i8, ptr %B, i64 16
  %6 = xor i32 %conv, -1
  %arrayidx393 = getelementptr inbounds nuw i8, ptr %B, i64 4
  %wide.trip.count741 = and i64 %call, 2147483647
  br label %while.body

for.body9:                                        ; preds = %for.end, %for.body9
  %indvars.iv = phi i64 [ 0, %for.end ], [ %indvars.iv.next, %for.body9 ]
  %arrayidx10 = getelementptr inbounds nuw [9 x i32], ptr %B, i64 0, i64 %indvars.iv
  store i32 %4, ptr %arrayidx10, align 4, !tbaa !9
  %arrayidx12 = getelementptr inbounds nuw [9 x i32], ptr %A, i64 0, i64 %indvars.iv
  store i32 %4, ptr %arrayidx12, align 4, !tbaa !9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond736.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond736.not, label %while.cond.preheader, label %for.body9, !llvm.loop !28

while.body:                                       ; preds = %while.body.lr.ph, %if.end424
  %call16734 = phi i32 [ %call16721, %while.body.lr.ph ], [ %call16, %if.end424 ]
  %j.0733 = phi i32 [ 0, %while.body.lr.ph ], [ %j.3.lcssa, %if.end424 ]
  %lasti.0732 = phi i32 [ 49152, %while.body.lr.ph ], [ %lasti.4, %if.end424 ]
  %tobool.not731 = phi i1 [ false, %while.body.lr.ph ], [ true, %if.end424 ]
  %B4.0730 = phi i32 [ %4, %while.body.lr.ph ], [ %B4.1.lcssa, %if.end424 ]
  %A4.0729 = phi i32 [ %4, %while.body.lr.ph ], [ %A4.1.lcssa, %if.end424 ]
  %B3.0728 = phi i32 [ %4, %while.body.lr.ph ], [ %B3.1.lcssa, %if.end424 ]
  %A3.0727 = phi i32 [ %4, %while.body.lr.ph ], [ %A3.1.lcssa, %if.end424 ]
  %B2.0726 = phi i32 [ %4, %while.body.lr.ph ], [ %B2.1.lcssa, %if.end424 ]
  %A2.0725 = phi i32 [ %4, %while.body.lr.ph ], [ %A2.1.lcssa, %if.end424 ]
  %B1.0724 = phi i32 [ %4, %while.body.lr.ph ], [ %B1.1.lcssa, %if.end424 ]
  %B0.0723 = phi i32 [ %4, %while.body.lr.ph ], [ %B0.1.lcssa, %if.end424 ]
  %add = add nuw nsw i32 %call16734, 49152
  br i1 %tobool.not731, label %if.end45, label %if.then19

if.then19:                                        ; preds = %while.body
  %7 = load i32, ptr @DELIMITER, align 4, !tbaa !9
  %tobool20.not = icmp eq i32 %7, 0
  br i1 %tobool20.not, label %if.end45, label %for.cond22.preheader

for.cond22.preheader:                             ; preds = %if.then19
  br i1 %cmp23688, label %for.end39, label %for.body25

for.cond22:                                       ; preds = %for.body25
  %indvars.iv.next738 = add nuw nsw i64 %indvars.iv737, 1
  %exitcond742.not = icmp eq i64 %indvars.iv.next738, %wide.trip.count741
  br i1 %exitcond742.not, label %for.end39, label %for.body25, !llvm.loop !29

for.body25:                                       ; preds = %for.cond22.preheader, %for.cond22
  %indvars.iv737 = phi i64 [ %indvars.iv.next738, %for.cond22 ], [ 0, %for.cond22.preheader ]
  %arrayidx27 = getelementptr inbounds nuw i8, ptr %old_D_pat, i64 %indvars.iv737
  %8 = load i8, ptr %arrayidx27, align 1, !tbaa !6
  %9 = add nuw nsw i64 %indvars.iv737, 49152
  %arrayidx31 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %9
  %10 = load i8, ptr %arrayidx31, align 1, !tbaa !6
  %cmp33.not = icmp eq i8 %8, %10
  br i1 %cmp33.not, label %for.cond22, label %for.end39

for.end39:                                        ; preds = %for.cond22, %for.body25, %for.cond22.preheader
  %cmp23.lcssa = phi i32 [ -1, %for.cond22.preheader ], [ 0, %for.body25 ], [ -1, %for.cond22 ]
  %spec.select = add nsw i32 %cmp23.lcssa, %j.0733
  br label %if.end45

if.end45:                                         ; preds = %for.end39, %if.then19, %while.body
  %i.1 = phi i32 [ 49152, %while.body ], [ 49151, %for.end39 ], [ 49151, %if.then19 ]
  %j.2 = phi i32 [ %j.0733, %while.body ], [ %spec.select, %for.end39 ], [ %j.0733, %if.then19 ]
  %cmp46 = icmp slt i32 %call16734, 49152
  br i1 %cmp46, label %if.then48, label %if.end57

if.then48:                                        ; preds = %if.end45
  %idx.ext = zext nneg i32 %add to i64
  %add.ptr50 = getelementptr inbounds nuw i8, ptr %buffer, i64 %idx.ext
  %call52 = call ptr @strncpy(ptr noundef nonnull %add.ptr50, ptr noundef nonnull %old_D_pat, i64 noundef %conv51) #6
  %add53 = add i32 %add, %conv
  %idxprom54 = zext i32 %add53 to i64
  %arrayidx55 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom54
  store i8 0, ptr %arrayidx55, align 1, !tbaa !6
  br label %if.end57

if.end57:                                         ; preds = %if.then48, %if.end45
  %end.0 = phi i32 [ %add53, %if.then48 ], [ %add, %if.end45 ]
  %cmp59699 = icmp ult i32 %i.1, %end.0
  br i1 %cmp59699, label %while.body61.lr.ph, label %while.end

while.body61.lr.ph:                               ; preds = %if.end57
  %sub160 = add nuw nsw i32 %call16734, 49151
  br label %while.body61

while.body61:                                     ; preds = %while.body61.lr.ph, %if.end400
  %j.3710 = phi i32 [ %j.2, %while.body61.lr.ph ], [ %j.5, %if.end400 ]
  %i.2709 = phi i32 [ %i.1, %while.body61.lr.ph ], [ %add288, %if.end400 ]
  %lasti.1708 = phi i32 [ %lasti.0732, %while.body61.lr.ph ], [ %lasti.3, %if.end400 ]
  %B4.1707 = phi i32 [ %B4.0730, %while.body61.lr.ph ], [ %B4.4, %if.end400 ]
  %A4.1706 = phi i32 [ %A4.0729, %while.body61.lr.ph ], [ %A4.4, %if.end400 ]
  %B3.1705 = phi i32 [ %B3.0728, %while.body61.lr.ph ], [ %B3.4, %if.end400 ]
  %A3.1704 = phi i32 [ %A3.0727, %while.body61.lr.ph ], [ %A3.4, %if.end400 ]
  %B2.1703 = phi i32 [ %B2.0726, %while.body61.lr.ph ], [ %B2.4, %if.end400 ]
  %A2.1702 = phi i32 [ %A2.0725, %while.body61.lr.ph ], [ %A2.4, %if.end400 ]
  %B1.1701 = phi i32 [ %B1.0724, %while.body61.lr.ph ], [ %B1.2, %if.end400 ]
  %B0.1700 = phi i32 [ %B0.0723, %while.body61.lr.ph ], [ %B0.2, %if.end400 ]
  %idxprom62 = zext i32 %i.2709 to i64
  %arrayidx63 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom62
  %11 = load i8, ptr %arrayidx63, align 1, !tbaa !6
  %idxprom65 = zext i8 %11 to i64
  %arrayidx66 = getelementptr inbounds nuw [0 x i32], ptr @Mask, i64 0, i64 %idxprom65
  %12 = load i32, ptr %arrayidx66, align 4, !tbaa !9
  %and = and i32 %B0.1700, %2
  %shr = lshr i32 %B0.1700, 1
  %and67 = and i32 %12, %shr
  %or68 = or i32 %and67, %and
  %and69 = and i32 %B1.1701, %2
  %or70 = or i32 %and67, %B0.1700
  %shr71 = lshr i32 %or70, 1
  %and72 = and i32 %shr71, %3
  %shr74 = lshr i32 %B1.1701, 1
  %and75 = and i32 %12, %shr74
  %13 = or i32 %and69, %and75
  %14 = or i32 %13, %and72
  %or77 = or i32 %14, %B0.1700
  br i1 %cmp78, label %Nextchar, label %if.end81

if.end81:                                         ; preds = %while.body61
  %and82 = and i32 %B2.1703, %2
  %or83 = or i32 %or77, %B1.1701
  %shr84 = lshr i32 %or83, 1
  %and85 = and i32 %shr84, %3
  %shr87 = lshr i32 %B2.1703, 1
  %and88 = and i32 %12, %shr87
  %15 = or i32 %and82, %and88
  %16 = or i32 %15, %and85
  %or90 = or i32 %16, %B1.1701
  br i1 %cmp91, label %Nextchar, label %if.end94

if.end94:                                         ; preds = %if.end81
  %and95 = and i32 %B3.1705, %2
  %or96 = or i32 %or90, %B2.1703
  %shr97 = lshr i32 %or96, 1
  %and98 = and i32 %shr97, %3
  %shr100 = lshr i32 %B3.1705, 1
  %and101 = and i32 %12, %shr100
  %17 = or i32 %and95, %and101
  %18 = or i32 %17, %and98
  %or103 = or i32 %18, %B2.1703
  br i1 %cmp104, label %Nextchar, label %if.end107

if.end107:                                        ; preds = %if.end94
  %and108 = and i32 %B4.1707, %2
  %or109 = or i32 %or103, %B3.1705
  %shr110 = lshr i32 %or109, 1
  %and111 = and i32 %shr110, %3
  %shr113 = lshr i32 %B4.1707, 1
  %and114 = and i32 %12, %shr113
  %19 = or i32 %and108, %and114
  %20 = or i32 %19, %and111
  %or116 = or i32 %20, %B3.1705
  br label %Nextchar

Nextchar:                                         ; preds = %if.end107, %if.end94, %if.end81, %while.body61
  %A2.2 = phi i32 [ %A2.1702, %while.body61 ], [ %or90, %if.end81 ], [ %or90, %if.end94 ], [ %or90, %if.end107 ]
  %A3.2 = phi i32 [ %A3.1704, %while.body61 ], [ %A3.1704, %if.end81 ], [ %or103, %if.end94 ], [ %or103, %if.end107 ]
  %A4.2 = phi i32 [ %A4.1706, %while.body61 ], [ %A4.1706, %if.end81 ], [ %A4.1706, %if.end94 ], [ %or116, %if.end107 ]
  %add121 = add nuw i32 %i.2709, 1
  %and122 = and i32 %or68, %1
  %tobool123.not = icmp eq i32 %and122, 0
  br i1 %tobool123.not, label %if.end226, label %if.then124

if.then124:                                       ; preds = %Nextchar
  %inc125 = add nsw i32 %j.3710, 1
  %spec.select661 = select i1 %cmp78, i32 %or77, i32 %or68
  %r1.1 = select i1 %cmp91, i32 %A2.2, i32 %spec.select661
  %r1.2 = select i1 %cmp104, i32 %A3.2, i32 %r1.1
  %r1.3 = select i1 %cmp138, i32 %A4.2, i32 %r1.2
  %21 = load i32, ptr @AND, align 4, !tbaa !9
  %cmp142 = icmp eq i32 %21, 1
  %.pre = load i32, ptr @endposition, align 4
  %and144 = and i32 %.pre, %r1.3
  %cmp145 = icmp eq i32 %and144, %.pre
  %or.cond = select i1 %cmp142, i1 %cmp145, i1 false
  br i1 %or.cond, label %if.then152, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.then124
  %cmp147 = icmp eq i32 %21, 0
  %tobool150 = icmp ne i32 %and144, 0
  %22 = select i1 %cmp147, i1 %tobool150, i1 false
  %land.ext = zext i1 %22 to i32
  %23 = load i32, ptr @INVERSE, align 4, !tbaa !9
  %tobool151.not = icmp eq i32 %23, %land.ext
  br i1 %tobool151.not, label %if.end166, label %if.then152

if.then152:                                       ; preds = %if.then124, %lor.lhs.false
  %24 = load i32, ptr @FILENAMEONLY, align 4, !tbaa !9
  %tobool153.not = icmp eq i32 %24, 0
  br i1 %tobool153.not, label %if.end157, label %if.then154

if.then154:                                       ; preds = %if.then152
  %25 = load i32, ptr @num_of_matched, align 4, !tbaa !9
  %inc155 = add nsw i32 %25, 1
  store i32 %inc155, ptr @num_of_matched, align 4, !tbaa !9
  %puts660 = call i32 @puts(ptr nonnull dereferenceable(1) @CurrentFileName)
  br label %cleanup

if.end157:                                        ; preds = %if.then152
  %cmp161.not = icmp slt i32 %lasti.1708, %sub160
  br i1 %cmp161.not, label %if.then163, label %if.end166

if.then163:                                       ; preds = %if.end157
  %sub158 = sub i32 %i.2709, %conv
  call void @output(ptr noundef nonnull %buffer, i32 noundef %lasti.1708, i32 noundef %sub158, i32 noundef %inc125) #6
  br label %if.end166

if.end166:                                        ; preds = %if.end157, %if.then163, %lor.lhs.false
  store i32 0, ptr @TRUNCATE, align 4, !tbaa !9
  %26 = load i32, ptr @Init, align 4, !tbaa !9
  br label %for.body171

for.body171:                                      ; preds = %if.end166, %for.body171
  %indvars.iv743 = phi i64 [ 0, %if.end166 ], [ %indvars.iv.next744, %for.body171 ]
  %arrayidx173 = getelementptr inbounds nuw [9 x i32], ptr %B, i64 0, i64 %indvars.iv743
  store i32 %26, ptr %arrayidx173, align 4, !tbaa !9
  %indvars.iv.next744 = add nuw nsw i64 %indvars.iv743, 1
  %exitcond747.not = icmp eq i64 %indvars.iv.next744, %wide.trip.count
  br i1 %exitcond747.not, label %for.end176, label %for.body171, !llvm.loop !30

for.end176:                                       ; preds = %for.body171
  %sub167 = sub i32 %add121, %conv
  %27 = load i32, ptr %B, align 4, !tbaa !9
  %28 = load i32, ptr @Init1, align 4, !tbaa !9
  %and178 = and i32 %28, %27
  %shr180 = lshr i32 %27, 1
  %and181 = and i32 %shr180, %12
  %or182 = or i32 %and181, %and178
  %and183 = and i32 %or182, %not
  store i32 %and183, ptr %A, align 4, !tbaa !9
  br i1 %cmp186.not694, label %for.end215, label %for.body188

for.body188:                                      ; preds = %for.end176, %for.body188
  %29 = phi i32 [ %or210, %for.body188 ], [ %and183, %for.end176 ]
  %30 = phi i32 [ %31, %for.body188 ], [ %27, %for.end176 ]
  %indvars.iv748 = phi i64 [ %indvars.iv.next749, %for.body188 ], [ 1, %for.end176 ]
  %arrayidx190 = getelementptr inbounds nuw [9 x i32], ptr %B, i64 0, i64 %indvars.iv748
  %31 = load i32, ptr %arrayidx190, align 4, !tbaa !9
  %and191 = and i32 %31, %28
  %or201 = or i32 %29, %30
  %shr202 = lshr i32 %or201, 1
  %and203 = and i32 %shr202, %3
  %shr207 = lshr i32 %31, 1
  %and208 = and i32 %shr207, %12
  %32 = or i32 %and191, %and208
  %33 = or i32 %32, %and203
  %or210 = or i32 %33, %30
  %arrayidx212 = getelementptr inbounds nuw [9 x i32], ptr %A, i64 0, i64 %indvars.iv748
  store i32 %or210, ptr %arrayidx212, align 4, !tbaa !9
  %indvars.iv.next749 = add nuw nsw i64 %indvars.iv748, 1
  %exitcond753.not = icmp eq i64 %indvars.iv.next749, %wide.trip.count
  br i1 %exitcond753.not, label %for.end215, label %for.body188, !llvm.loop !31

for.end215:                                       ; preds = %for.body188, %for.end176
  %34 = load i32, ptr %arrayidx218, align 4, !tbaa !9
  %35 = load i32, ptr %arrayidx220, align 4, !tbaa !9
  %36 = load i32, ptr %arrayidx221, align 4, !tbaa !9
  %37 = load i32, ptr %arrayidx222, align 4, !tbaa !9
  %38 = load i32, ptr %arrayidx223, align 4, !tbaa !9
  %39 = load i32, ptr %arrayidx224, align 4, !tbaa !9
  %40 = load i32, ptr %arrayidx225, align 4, !tbaa !9
  br label %if.end226

if.end226:                                        ; preds = %for.end215, %Nextchar
  %A0.0 = phi i32 [ %and183, %for.end215 ], [ %or68, %Nextchar ]
  %A1.0 = phi i32 [ %34, %for.end215 ], [ %or77, %Nextchar ]
  %A2.3 = phi i32 [ %35, %for.end215 ], [ %A2.2, %Nextchar ]
  %B2.2 = phi i32 [ %36, %for.end215 ], [ %B2.1703, %Nextchar ]
  %A3.3 = phi i32 [ %37, %for.end215 ], [ %A3.2, %Nextchar ]
  %B3.2 = phi i32 [ %38, %for.end215 ], [ %B3.1705, %Nextchar ]
  %A4.3 = phi i32 [ %39, %for.end215 ], [ %A4.2, %Nextchar ]
  %B4.2 = phi i32 [ %40, %for.end215 ], [ %B4.1707, %Nextchar ]
  %lasti.2 = phi i32 [ %sub167, %for.end215 ], [ %lasti.1708, %Nextchar ]
  %j.4 = phi i32 [ %inc125, %for.end215 ], [ %j.3710, %Nextchar ]
  %idxprom227 = zext i32 %add121 to i64
  %arrayidx228 = getelementptr inbounds nuw [98305 x i8], ptr %buffer, i64 0, i64 %idxprom227
  %41 = load i8, ptr %arrayidx228, align 1, !tbaa !6
  %idxprom230 = zext i8 %41 to i64
  %arrayidx231 = getelementptr inbounds nuw [0 x i32], ptr @Mask, i64 0, i64 %idxprom230
  %42 = load i32, ptr %arrayidx231, align 4, !tbaa !9
  %and232 = and i32 %A0.0, %2
  %shr233 = lshr i32 %A0.0, 1
  %and234 = and i32 %42, %shr233
  %or235 = or i32 %and234, %and232
  %and236 = and i32 %A1.0, %2
  %or237 = or i32 %and234, %A0.0
  %shr238 = lshr i32 %or237, 1
  %and239 = and i32 %shr238, %3
  %shr241 = lshr i32 %A1.0, 1
  %and242 = and i32 %42, %shr241
  %43 = or i32 %and236, %and242
  %44 = or i32 %43, %and239
  %or244 = or i32 %44, %A0.0
  br i1 %cmp78, label %Nextchar1, label %if.end248

if.end248:                                        ; preds = %if.end226
  %and249 = and i32 %A2.3, %2
  %or250 = or i32 %or244, %A1.0
  %shr251 = lshr i32 %or250, 1
  %and252 = and i32 %shr251, %3
  %shr254 = lshr i32 %A2.3, 1
  %and255 = and i32 %42, %shr254
  %45 = or i32 %and249, %and255
  %46 = or i32 %45, %and252
  %or257 = or i32 %46, %A1.0
  br i1 %cmp91, label %Nextchar1, label %if.end261

if.end261:                                        ; preds = %if.end248
  %and262 = and i32 %A3.3, %2
  %or263 = or i32 %or257, %A2.3
  %shr264 = lshr i32 %or263, 1
  %and265 = and i32 %shr264, %3
  %shr267 = lshr i32 %A3.3, 1
  %and268 = and i32 %42, %shr267
  %47 = or i32 %and262, %and268
  %48 = or i32 %47, %and265
  %or270 = or i32 %48, %A2.3
  br i1 %cmp104, label %Nextchar1, label %if.end274

if.end274:                                        ; preds = %if.end261
  %and275 = and i32 %A4.3, %2
  %or276 = or i32 %or270, %A3.3
  %shr277 = lshr i32 %or276, 1
  %and278 = and i32 %shr277, %3
  %shr280 = lshr i32 %A4.3, 1
  %and281 = and i32 %42, %shr280
  %49 = or i32 %and275, %and281
  %50 = or i32 %49, %and278
  %or283 = or i32 %50, %A3.3
  br label %Nextchar1

Nextchar1:                                        ; preds = %if.end274, %if.end261, %if.end248, %if.end226
  %B2.3 = phi i32 [ %B2.2, %if.end226 ], [ %or257, %if.end248 ], [ %or257, %if.end261 ], [ %or257, %if.end274 ]
  %B3.3 = phi i32 [ %B3.2, %if.end226 ], [ %B3.2, %if.end248 ], [ %or270, %if.end261 ], [ %or270, %if.end274 ]
  %B4.3 = phi i32 [ %B4.2, %if.end226 ], [ %B4.2, %if.end248 ], [ %B4.2, %if.end261 ], [ %or283, %if.end274 ]
  %add288 = add i32 %i.2709, 2
  %and289 = and i32 %or235, %1
  %tobool290.not = icmp eq i32 %and289, 0
  br i1 %tobool290.not, label %if.end400, label %if.then291

if.then291:                                       ; preds = %Nextchar1
  %inc292 = add nsw i32 %j.4, 1
  %spec.select662 = select i1 %cmp78, i32 %or244, i32 %or235
  %r1.5 = select i1 %cmp91, i32 %B2.3, i32 %spec.select662
  %r1.6 = select i1 %cmp104, i32 %B3.3, i32 %r1.5
  %r1.7 = select i1 %cmp138, i32 %B4.3, i32 %r1.6
  %51 = load i32, ptr @AND, align 4, !tbaa !9
  %cmp309 = icmp eq i32 %51, 1
  %.pre765 = load i32, ptr @endposition, align 4
  %and312 = and i32 %.pre765, %r1.7
  %cmp313 = icmp eq i32 %and312, %.pre765
  %or.cond768 = select i1 %cmp309, i1 %cmp313, i1 false
  br i1 %or.cond768, label %if.then325, label %lor.lhs.false315

lor.lhs.false315:                                 ; preds = %if.then291
  %cmp316 = icmp eq i32 %51, 0
  %tobool320 = icmp ne i32 %and312, 0
  %52 = select i1 %cmp316, i1 %tobool320, i1 false
  %land.ext322 = zext i1 %52 to i32
  %53 = load i32, ptr @INVERSE, align 4, !tbaa !9
  %tobool324.not = icmp eq i32 %53, %land.ext322
  br i1 %tobool324.not, label %if.end340, label %if.then325

if.then325:                                       ; preds = %if.then291, %lor.lhs.false315
  %54 = load i32, ptr @FILENAMEONLY, align 4, !tbaa !9
  %tobool326.not = icmp eq i32 %54, 0
  br i1 %tobool326.not, label %if.end330, label %if.then327

if.then327:                                       ; preds = %if.then325
  %55 = load i32, ptr @num_of_matched, align 4, !tbaa !9
  %inc328 = add nsw i32 %55, 1
  store i32 %inc328, ptr @num_of_matched, align 4, !tbaa !9
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @CurrentFileName)
  br label %cleanup

if.end330:                                        ; preds = %if.then325
  %cmp335.not = icmp slt i32 %lasti.2, %sub160
  br i1 %cmp335.not, label %if.then337, label %if.end340

if.then337:                                       ; preds = %if.end330
  %sub332 = add i32 %add288, %6
  call void @output(ptr noundef nonnull %buffer, i32 noundef %lasti.2, i32 noundef %sub332, i32 noundef %inc292) #6
  br label %if.end340

if.end340:                                        ; preds = %if.end330, %if.then337, %lor.lhs.false315
  store i32 0, ptr @TRUNCATE, align 4, !tbaa !9
  %56 = load i32, ptr @Init, align 4, !tbaa !9
  br label %for.body345

for.body345:                                      ; preds = %if.end340, %for.body345
  %indvars.iv754 = phi i64 [ 0, %if.end340 ], [ %indvars.iv.next755, %for.body345 ]
  %arrayidx347 = getelementptr inbounds nuw [9 x i32], ptr %A, i64 0, i64 %indvars.iv754
  store i32 %56, ptr %arrayidx347, align 4, !tbaa !9
  %indvars.iv.next755 = add nuw nsw i64 %indvars.iv754, 1
  %exitcond758.not = icmp eq i64 %indvars.iv.next755, %wide.trip.count
  br i1 %exitcond758.not, label %for.end350, label %for.body345, !llvm.loop !32

for.end350:                                       ; preds = %for.body345
  %sub341 = sub i32 %add288, %conv
  %57 = load i32, ptr %A, align 4, !tbaa !9
  %58 = load i32, ptr @Init1, align 4, !tbaa !9
  %and352 = and i32 %58, %57
  %shr354 = lshr i32 %57, 1
  %and355 = and i32 %shr354, %42
  %or356 = or i32 %and355, %and352
  %and357 = and i32 %or356, %not
  store i32 %and357, ptr %B, align 4, !tbaa !9
  br i1 %cmp186.not694, label %for.end389, label %for.body362

for.body362:                                      ; preds = %for.end350, %for.body362
  %59 = phi i32 [ %or384, %for.body362 ], [ %and357, %for.end350 ]
  %60 = phi i32 [ %61, %for.body362 ], [ %57, %for.end350 ]
  %indvars.iv759 = phi i64 [ %indvars.iv.next760, %for.body362 ], [ 1, %for.end350 ]
  %arrayidx364 = getelementptr inbounds nuw [9 x i32], ptr %A, i64 0, i64 %indvars.iv759
  %61 = load i32, ptr %arrayidx364, align 4, !tbaa !9
  %and365 = and i32 %61, %58
  %or375 = or i32 %59, %60
  %shr376 = lshr i32 %or375, 1
  %and377 = and i32 %shr376, %3
  %shr381 = lshr i32 %61, 1
  %and382 = and i32 %shr381, %42
  %62 = or i32 %and365, %and382
  %63 = or i32 %62, %and377
  %or384 = or i32 %63, %60
  %arrayidx386 = getelementptr inbounds nuw [9 x i32], ptr %B, i64 0, i64 %indvars.iv759
  store i32 %or384, ptr %arrayidx386, align 4, !tbaa !9
  %indvars.iv.next760 = add nuw nsw i64 %indvars.iv759, 1
  %exitcond764.not = icmp eq i64 %indvars.iv.next760, %wide.trip.count
  br i1 %exitcond764.not, label %for.end389, label %for.body362, !llvm.loop !33

for.end389:                                       ; preds = %for.body362, %for.end350
  %64 = load i32, ptr %arrayidx393, align 4, !tbaa !9
  %65 = load i32, ptr %arrayidx220, align 4, !tbaa !9
  %66 = load i32, ptr %arrayidx221, align 4, !tbaa !9
  %67 = load i32, ptr %arrayidx222, align 4, !tbaa !9
  %68 = load i32, ptr %arrayidx223, align 4, !tbaa !9
  %69 = load i32, ptr %arrayidx224, align 4, !tbaa !9
  %70 = load i32, ptr %arrayidx225, align 4, !tbaa !9
  br label %if.end400

if.end400:                                        ; preds = %for.end389, %Nextchar1
  %B0.2 = phi i32 [ %and357, %for.end389 ], [ %or235, %Nextchar1 ]
  %B1.2 = phi i32 [ %64, %for.end389 ], [ %or244, %Nextchar1 ]
  %A2.4 = phi i32 [ %65, %for.end389 ], [ %A2.3, %Nextchar1 ]
  %B2.4 = phi i32 [ %66, %for.end389 ], [ %B2.3, %Nextchar1 ]
  %A3.4 = phi i32 [ %67, %for.end389 ], [ %A3.3, %Nextchar1 ]
  %B3.4 = phi i32 [ %68, %for.end389 ], [ %B3.3, %Nextchar1 ]
  %A4.4 = phi i32 [ %69, %for.end389 ], [ %A4.3, %Nextchar1 ]
  %B4.4 = phi i32 [ %70, %for.end389 ], [ %B4.3, %Nextchar1 ]
  %lasti.3 = phi i32 [ %sub341, %for.end389 ], [ %lasti.2, %Nextchar1 ]
  %j.5 = phi i32 [ %inc292, %for.end389 ], [ %j.4, %Nextchar1 ]
  %cmp59 = icmp ult i32 %add288, %end.0
  br i1 %cmp59, label %while.body61, label %while.end, !llvm.loop !34

while.end:                                        ; preds = %if.end400, %if.end57
  %B0.1.lcssa = phi i32 [ %B0.0723, %if.end57 ], [ %B0.2, %if.end400 ]
  %B1.1.lcssa = phi i32 [ %B1.0724, %if.end57 ], [ %B1.2, %if.end400 ]
  %A2.1.lcssa = phi i32 [ %A2.0725, %if.end57 ], [ %A2.4, %if.end400 ]
  %B2.1.lcssa = phi i32 [ %B2.0726, %if.end57 ], [ %B2.4, %if.end400 ]
  %A3.1.lcssa = phi i32 [ %A3.0727, %if.end57 ], [ %A3.4, %if.end400 ]
  %B3.1.lcssa = phi i32 [ %B3.0728, %if.end57 ], [ %B3.4, %if.end400 ]
  %A4.1.lcssa = phi i32 [ %A4.0729, %if.end57 ], [ %A4.4, %if.end400 ]
  %B4.1.lcssa = phi i32 [ %B4.0730, %if.end57 ], [ %B4.4, %if.end400 ]
  %lasti.1.lcssa = phi i32 [ %lasti.0732, %if.end57 ], [ %lasti.3, %if.end400 ]
  %j.3.lcssa = phi i32 [ %j.2, %if.end57 ], [ %j.5, %if.end400 ]
  br i1 %cmp46, label %if.end424, label %if.else

if.else:                                          ; preds = %while.end
  %sub405 = sub nsw i32 %add, %lasti.1.lcssa
  %cmp406 = icmp sgt i32 %sub405, 49152
  br i1 %cmp406, label %if.then408, label %if.end409

if.then408:                                       ; preds = %if.else
  store i32 1, ptr @TRUNCATE, align 4, !tbaa !9
  br label %if.end409

if.end409:                                        ; preds = %if.then408, %if.else
  %ResidueSize.0 = phi i32 [ 49152, %if.then408 ], [ %sub405, %if.else ]
  %ResidueSize.0.fr = freeze i32 %ResidueSize.0
  %idx.ext412 = sext i32 %ResidueSize.0.fr to i64
  %idx.neg = sub nsw i64 0, %idx.ext412
  %add.ptr413 = getelementptr inbounds i8, ptr %add.ptr, i64 %idx.neg
  %idx.ext415 = sext i32 %lasti.1.lcssa to i64
  %add.ptr416 = getelementptr inbounds i8, ptr %buffer, i64 %idx.ext415
  %call418 = call ptr @strncpy(ptr noundef nonnull %add.ptr413, ptr noundef nonnull %add.ptr416, i64 noundef %idx.ext412) #6
  %sub419 = sub nsw i32 49152, %ResidueSize.0.fr
  %cmp420 = icmp eq i32 %ResidueSize.0.fr, 49152
  %spec.select663 = select i1 %cmp420, i32 1, i32 %sub419
  br label %if.end424

if.end424:                                        ; preds = %if.end409, %while.end
  %lasti.4 = phi i32 [ 49152, %while.end ], [ %spec.select663, %if.end409 ]
  %call16 = call i32 @fill_buf(i32 noundef %text, ptr noundef nonnull %add.ptr, i32 noundef 49152) #6
  %cmp17 = icmp sgt i32 %call16, 0
  br i1 %cmp17, label %while.body, label %cleanup, !llvm.loop !35

cleanup:                                          ; preds = %if.end424, %while.cond.preheader, %if.then327, %if.then154, %if.then2
  call void @llvm.lifetime.end.p0(ptr nonnull %buffer) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %B) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %A) #6
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #5

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind }
attributes #6 = { nounwind }
attributes #7 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 34109cd26ae1b317d91c061500d9828fe6ebab0b)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12, !14, !15}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !12, !15, !14}
!17 = distinct !{!17, !12}
!18 = distinct !{!18, !12, !14, !15}
!19 = distinct !{!19, !12, !15, !14}
!20 = distinct !{!20, !12}
!21 = distinct !{!21, !12}
!22 = distinct !{!22, !12, !14, !15}
!23 = distinct !{!23, !12, !15, !14}
!24 = distinct !{!24, !12}
!25 = distinct !{!25, !12}
!26 = distinct !{!26, !12}
!27 = distinct !{!27, !12}
!28 = distinct !{!28, !12, !15, !14}
!29 = distinct !{!29, !12}
!30 = distinct !{!30, !12, !15, !14}
!31 = distinct !{!31, !12}
!32 = distinct !{!32, !12, !15, !14}
!33 = distinct !{!33, !12}
!34 = distinct !{!34, !12}
!35 = distinct !{!35, !12}
