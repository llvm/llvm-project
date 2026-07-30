// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTRY   | FileCheck %s -check-prefix=TRY
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTHROW | FileCheck %s -check-prefix=THROW

// THROW-DAG: @"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat
// THROW-DAG: @"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, ptr @"??_R0H@8", i32 0, i32 -1, i32 0, i32 4, ptr null }, section ".xdata", comdat
// THROW-DAG: @_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x ptr] [ptr @"_CT??_R0H@84"] }, section ".xdata", comdat
// THROW-DAG: @_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, ptr null, ptr null, ptr @_CTA1H }, section ".xdata", comdat

void external();

inline void not_emitted() {
  throw int(13); // no error
}

int main() {
  int rv = 0;
#ifdef TRY
  try {
    external(); // TRY: invoke void @"?external@@YAXXZ"
  } catch (int) {
    rv = 1;
    // TRY: catchpad within {{.*}} [ptr @"??_R0H@8", i32 0, ptr null]
    // TRY: catchret
  }
#endif
#ifdef THROW
  // THROW: store i32 42, ptr %[[mem_for_throw:.*]], align 4
  // THROW: call void @_CxxThrowException(ptr %[[mem_for_throw]], ptr @_TI1H)
  throw int(42);
#endif
  return rv;
}

#ifdef TRY
// TRY-LABEL: define dso_local void @"?qual_catch@@YAXXZ"
void qual_catch() {
  try {
    external();
  } catch (const int *) {
  }
  // TRY: catchpad within {{.*}} [ptr @"??_R0PAH@8", i32 1, ptr null]
  // TRY: catchret
}
#endif
