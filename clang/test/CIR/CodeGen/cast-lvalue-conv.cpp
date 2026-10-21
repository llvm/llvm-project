// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test CK_ConstructorConversion and CK_UserDefinedConversion
// in emitCastLValue (pass-through to sub-expression).

struct Source {
  int val;
  operator int() const { return val; }
};

struct Dest {
  int val;
  Dest(const Source &s) : val(s.val) {}
};

// CK_ConstructorConversion: implicit conversion via constructor
// when binding a reference.
void takeDestRef(const Dest &d);

void testConstructorConversion() {
  Source s{42};
  takeDestRef(s);
}

// CIR-LABEL: cir.func {{.*}} @_Z25testConstructorConversionv
// CIR:   cir.call @_ZN4DestC1ERK6Source
// CIR:   cir.call @_Z11takeDestRefRK4Dest

// LLVM-LABEL: define {{.*}} @_Z25testConstructorConversionv
// LLVM:   call void @_ZN4DestC1ERK6Source
// LLVM:   call void @_Z11takeDestRefRK4Dest

// OGCG-LABEL: define {{.*}} @_Z25testConstructorConversionv
// OGCG:   call void @_ZN4DestC1ERK6Source
// OGCG:   call void @_Z11takeDestRefRK4Dest

// CK_UserDefinedConversion: implicit conversion via operator.
void takeInt(int x);

int testUserDefinedConversion() {
  Source s{7};
  return static_cast<int>(s);
}

// CIR-LABEL: cir.func {{.*}} @_Z25testUserDefinedConversionv
// CIR:   cir.call @_ZNK6SourcecviEv

// LLVM-LABEL: define {{.*}} @_Z25testUserDefinedConversionv
// LLVM:   call noundef i32 @_ZNK6SourcecviEv

// OGCG-LABEL: define {{.*}} @_Z25testUserDefinedConversionv
// OGCG:   call noundef i32 @_ZNK6SourcecviEv
