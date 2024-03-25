// RUN: %clang_cc1 -ffixed-point -S -emit-llvm %s -o - -triple=x86_64-unknown-linux-gnu | FileCheck %s

// Primary fixed point types
void func(signed short _Accum){}    // CHECK: @_Z4funcDAs
void func(signed _Accum){}          // CHECK: @_Z4funcDAi
void func(signed long _Accum){}     // CHECK: @_Z4funcDAl
void func(unsigned short _Accum){}  // CHECK: @_Z4funcDAt
void func(unsigned _Accum){}        // CHECK: @_Z4funcDAj
void func(unsigned long _Accum){}   // CHECK: @_Z4funcDAm
void func(signed short _Fract){}    // CHECK: @_Z4funcDRs
void func(signed _Fract){}          // CHECK: @_Z4funcDRi
void func(signed long _Fract){}     // CHECK: @_Z4funcDRl
void func(unsigned short _Fract){}  // CHECK: @_Z4funcDRt
void func(unsigned _Fract){}        // CHECK: @_Z4funcDRj
void func(unsigned long _Fract){}   // CHECK: @_Z4funcDRm

// Aliased
void func2(short _Accum){}          // CHECK: @_Z5func2DAs
void func2(_Accum){}                // CHECK: @_Z5func2DAi
void func2(long _Accum){}           // CHECK: @_Z5func2DAl
void func2(short _Fract){}          // CHECK: @_Z5func2DRs
void func2(_Fract){}                // CHECK: @_Z5func2DRi
void func2(long _Fract){}           // CHECK: @_Z5func2DRl

// Primary saturated
void func(_Sat signed short _Accum){}    // CHECK: @_Z4funcDSDAs
void func(_Sat signed _Accum){}          // CHECK: @_Z4funcDSDAi
void func(_Sat signed long _Accum){}     // CHECK: @_Z4funcDSDAl
void func(_Sat unsigned short _Accum){}  // CHECK: @_Z4funcDSDAt
void func(_Sat unsigned _Accum){}        // CHECK: @_Z4funcDSDAj
void func(_Sat unsigned long _Accum){}   // CHECK: @_Z4funcDSDAm
void func(_Sat signed short _Fract){}    // CHECK: @_Z4funcDSDRs
void func(_Sat signed _Fract){}          // CHECK: @_Z4funcDSDRi
void func(_Sat signed long _Fract){}     // CHECK: @_Z4funcDSDRl
void func(_Sat unsigned short _Fract){}  // CHECK: @_Z4funcDSDRt
void func(_Sat unsigned _Fract){}        // CHECK: @_Z4funcDSDRj
void func(_Sat unsigned long _Fract){}   // CHECK: @_Z4funcDSDRm

// Aliased saturated
void func2(_Sat short _Accum){}          // CHECK: @_Z5func2DSDAs
void func2(_Sat _Accum){}                // CHECK: @_Z5func2DSDAi
void func2(_Sat long _Accum){}           // CHECK: @_Z5func2DSDAl
void func2(_Sat short _Fract){}          // CHECK: @_Z5func2DSDRs
void func2(_Sat _Fract){}                // CHECK: @_Z5func2DSDRi
void func2(_Sat long _Fract){}           // CHECK: @_Z5func2DSDRl
