// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

// CHECK:  @_ZZ4FUNCvEN4SSSSC1ERKf
// CHECK: @_ZZ4FUNCvEN4SSSSC2E_0RKf
// CHECK:  @_ZZ4GORFfEN4SSSSC1ERKf
// CHECK: @_ZZ4GORFfEN4SSSSC2E_0RKf

void FUNC ()
{
  {
    float IVAR1 ;

    struct SSSS 
    {
      float bv;
      SSSS( const float& from): bv(from) { }
    };

    SSSS VAR1(IVAR1);
   }

   {
    float IVAR2 ;

    struct SSSS
    {
     SSSS( const float& from) {}
    };

    SSSS VAR2(IVAR2);
   }
}

void GORF (float IVAR1)
{
  {
    struct SSSS 
    {
      float bv;
      SSSS( const float& from): bv(from) { }
    };

    SSSS VAR1(IVAR1);
   }

   {
    float IVAR2 ;

    struct SSSS
    {
     SSSS( const float& from) {}
    };

    SSSS VAR2(IVAR2);
   }
}

// CHECK: @_ZZ12OmittingCodefEN4SSSSC1E_0RKf
inline void OmittingCode(float x) {
  if (0) {
    struct SSSS {
      float bv;
      SSSS(const float& from): bv(from) { }
    };

    SSSS VAR1(x);
  }

  struct SSSS {
    float bv;
    SSSS(const float& from): bv(from) { }
  };

  SSSS VAR2(x);
}
void CallOmittingCode() { OmittingCode(1); }

// CHECK: @_ZZ15LocalAnonStructvENUt0_1gEv
inline void LocalAnonStruct() {
  if (0) {
    struct { void f() {} } x;
    x.f();
  }
  struct { void g() {} } y;
  y.g();
}
void CallLocalAnonStruct() { LocalAnonStruct(); }
