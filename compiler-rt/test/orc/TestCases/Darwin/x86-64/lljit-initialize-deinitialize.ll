; RUN: %lli_orc_jitlink -relocation-model=pic %s | FileCheck %s

; CHECK: constructor
; CHECK-NEXT: main
; CHECK-NEXT: destructor

@__dso_handle = external hidden global i8
@.str = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"constructor\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"destructor\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @constructor, i8* null }]

define dso_local void @destructor(i8* %0) {
  %2 = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([11 x i8], [11 x i8]* @.str.2, i64 0, i64 0))
  ret void
}

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local i32 @main(i32 %0, i8** nocapture readnone %1) local_unnamed_addr #2 {
  %3 = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture readonly)

define internal void @constructor() {
  %1 = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0)) #5
  %2 = tail call i32 @__cxa_atexit(void (i8*)* @destructor, i8* null, i8* nonnull @__dso_handle) #5
  ret void
}
