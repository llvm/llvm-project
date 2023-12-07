// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm-only -fdump-vtable-layouts 2>&1 | FileCheck %s

// A collection of big class hierarchies and their vtables.

namespace Test1 {

class C0
{
};
class C1
 :  virtual public C0
{
  int k0;
};
class C2
 :  public C0
 ,  virtual public C1
{
  int k0;
};
class C3
 :  virtual public C0
 ,  virtual public C1
 ,  public C2
{
  int k0;
  int k1;
  int k2;
  int k3;
};
class C4
 :  public C2
 ,  virtual public C3
 ,  public C0
{
  int k0;
};
class C5
 :  public C0
 ,  virtual public C4
 ,  public C2
 ,  public C1
 ,  virtual public C3
{
  int k0;
};
class C6
 :  virtual public C3
 ,  public C0
 ,  public C5
 ,  public C4
 ,  public C1
{
  int k0;
};
class C7
 :  virtual public C5
 ,  virtual public C6
 ,  virtual public C3
 ,  public C4
 ,  virtual public C2
{
  int k0;
  int k1;
};
class C8
 :  public C7
 ,  public C5
 ,  public C3
 ,  virtual public C4
 ,  public C1
 ,  public C2
{
  int k0;
  int k1;
};

// CHECK:     Vtable for 'Test1::C9' (87 entries).
// CHECK-NEXT:   0 | vbase_offset (344)
// CHECK-NEXT:   1 | vbase_offset (312)
// CHECK-NEXT:   2 | vbase_offset (184)
// CHECK-NEXT:   3 | vbase_offset (168)
// CHECK-NEXT:   4 | vbase_offset (120)
// CHECK-NEXT:   5 | vbase_offset (48)
// CHECK-NEXT:   6 | vbase_offset (148)
// CHECK-NEXT:   7 | vbase_offset (152)
// CHECK-NEXT:   8 | offset_to_top (0)
// CHECK-NEXT:   9 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 0) vtable address --
// CHECK-NEXT:       -- (Test1::C9, 0) vtable address --
// CHECK-NEXT:  10 | void Test1::C9::f()
// CHECK-NEXT:  11 | vbase_offset (104)
// CHECK-NEXT:  12 | vbase_offset (132)
// CHECK-NEXT:  13 | vbase_offset (136)
// CHECK-NEXT:  14 | offset_to_top (-16)
// CHECK-NEXT:  15 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 16) vtable address --
// CHECK-NEXT:       -- (Test1::C4, 16) vtable address --
// CHECK-NEXT:  16 | vbase_offset (72)
// CHECK-NEXT:  17 | vbase_offset (120)
// CHECK-NEXT:  18 | vbase_offset (100)
// CHECK-NEXT:  19 | vbase_offset (104)
// CHECK-NEXT:  20 | offset_to_top (-48)
// CHECK-NEXT:  21 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 48) vtable address --
// CHECK-NEXT:       -- (Test1::C5, 48) vtable address --
// CHECK-NEXT:       -- (Test1::C6, 48) vtable address --
// CHECK-NEXT:  22 | vbase_offset (84)
// CHECK-NEXT:  23 | offset_to_top (-64)
// CHECK-NEXT:  24 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 64) vtable address --
// CHECK-NEXT:  25 | vbase_offset (32)
// CHECK-NEXT:  26 | vbase_offset (60)
// CHECK-NEXT:  27 | vbase_offset (64)
// CHECK-NEXT:  28 | offset_to_top (-88)
// CHECK-NEXT:  29 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 88) vtable address --
// CHECK-NEXT:       -- (Test1::C4, 88) vtable address --
// CHECK-NEXT:  30 | vbase_offset (44)
// CHECK-NEXT:  31 | offset_to_top (-104)
// CHECK-NEXT:  32 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 104) vtable address --
// CHECK-NEXT:  33 | vbase_offset (28)
// CHECK-NEXT:  34 | vbase_offset (32)
// CHECK-NEXT:  35 | offset_to_top (-120)
// CHECK-NEXT:  36 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 120) vtable address --
// CHECK-NEXT:       -- (Test1::C3, 120) vtable address --
// CHECK-NEXT:  37 | vbase_offset (-4)
// CHECK-NEXT:  38 | offset_to_top (-152)
// CHECK-NEXT:  39 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 152) vtable address --
// CHECK-NEXT:  40 | vbase_offset (-48)
// CHECK-NEXT:  41 | vbase_offset (-20)
// CHECK-NEXT:  42 | vbase_offset (-16)
// CHECK-NEXT:  43 | offset_to_top (-168)
// CHECK-NEXT:  44 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 168) vtable address --
// CHECK-NEXT:       -- (Test1::C4, 168) vtable address --
// CHECK-NEXT:  45 | vbase_offset (160)
// CHECK-NEXT:  46 | vbase_offset (-136)
// CHECK-NEXT:  47 | vbase_offset (-16)
// CHECK-NEXT:  48 | vbase_offset (128)
// CHECK-NEXT:  49 | vbase_offset (-64)
// CHECK-NEXT:  50 | vbase_offset (-36)
// CHECK-NEXT:  51 | vbase_offset (-32)
// CHECK-NEXT:  52 | offset_to_top (-184)
// CHECK-NEXT:  53 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 184) vtable address --
// CHECK-NEXT:       -- (Test1::C4, 184) vtable address --
// CHECK-NEXT:       -- (Test1::C7, 184) vtable address --
// CHECK-NEXT:       -- (Test1::C8, 184) vtable address --
// CHECK-NEXT:  54 | vbase_offset (-88)
// CHECK-NEXT:  55 | vbase_offset (-40)
// CHECK-NEXT:  56 | vbase_offset (-60)
// CHECK-NEXT:  57 | vbase_offset (-56)
// CHECK-NEXT:  58 | offset_to_top (-208)
// CHECK-NEXT:  59 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 208) vtable address --
// CHECK-NEXT:       -- (Test1::C5, 208) vtable address --
// CHECK-NEXT:  60 | vbase_offset (-76)
// CHECK-NEXT:  61 | offset_to_top (-224)
// CHECK-NEXT:  62 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 224) vtable address --
// CHECK-NEXT:  63 | vbase_offset (-92)
// CHECK-NEXT:  64 | vbase_offset (-88)
// CHECK-NEXT:  65 | offset_to_top (-240)
// CHECK-NEXT:  66 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 240) vtable address --
// CHECK-NEXT:       -- (Test1::C3, 240) vtable address --
// CHECK-NEXT:  67 | vbase_offset (-124)
// CHECK-NEXT:  68 | offset_to_top (-272)
// CHECK-NEXT:  69 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 272) vtable address --
// CHECK-NEXT:  70 | vbase_offset (-140)
// CHECK-NEXT:  71 | vbase_offset (-136)
// CHECK-NEXT:  72 | offset_to_top (-288)
// CHECK-NEXT:  73 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 288) vtable address --
// CHECK-NEXT:  74 | vbase_offset (-192)
// CHECK-NEXT:  75 | vbase_offset (-144)
// CHECK-NEXT:  76 | vbase_offset (-164)
// CHECK-NEXT:  77 | vbase_offset (-160)
// CHECK-NEXT:  78 | offset_to_top (-312)
// CHECK-NEXT:  79 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C2, 312) vtable address --
// CHECK-NEXT:       -- (Test1::C5, 312) vtable address --
// CHECK-NEXT:  80 | vbase_offset (-180)
// CHECK-NEXT:  81 | offset_to_top (-328)
// CHECK-NEXT:  82 | Test1::C9 RTTI
// CHECK-NEXT:       -- (Test1::C1, 328) vtable address --
// CHECK-NEXT:  83 | vbase_offset (-196)
// CHECK-NEXT:  84 | vbase_offset (-192)
// CHECK-NEXT:  85 | offset_to_top (-344)
// CHECK-NEXT:  86 | Test1::C9 RTTI
class C9
 :  virtual public C6
 ,  public C2
 ,  public C4
 ,  virtual public C8
{
  int k0;
  int k1;
  int k2;
  int k3;
  virtual void f();
};
void C9::f() { }

}
