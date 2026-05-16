//===-- BuiltinsSystemZ.td - SystemZ Builtin function database --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SystemZ-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def SystemZPrefix : NamePrefix<"__builtin_s390_">;

class SystemZTargetBuiltin<string prototype> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let RequiredNamePrefix = SystemZPrefix;
}

class SystemZNoPrefixTargetBuiltin<string prototype> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
}

// Transactional-memory intrinsics
let Features = "transactional-execution" in {
  let Attributes = [ReturnsTwice] in {
    def __builtin_tbegin : SystemZNoPrefixTargetBuiltin<"int(void *)">;
    def __builtin_tbegin_nofloat : SystemZNoPrefixTargetBuiltin<"int(void *)">;
  }
  let Attributes = [NoThrow, ReturnsTwice] in
    def __builtin_tbeginc : SystemZNoPrefixTargetBuiltin<"void()">;
  let Attributes = [NoReturn] in
    def __builtin_tabort : SystemZNoPrefixTargetBuiltin<"void(int)">;
  let Attributes = [NoThrow] in
    def __builtin_tend : SystemZNoPrefixTargetBuiltin<"int()">;
  let Attributes = [NoThrow, Const] in
    def __builtin_tx_nesting_depth : SystemZNoPrefixTargetBuiltin<"int()">;
  let Attributes = [NoThrow] in
    def __builtin_tx_assist : SystemZNoPrefixTargetBuiltin<"void(int)">;
  def __builtin_non_tx_store : SystemZNoPrefixTargetBuiltin<"void(unsigned long int *, unsigned long int)">;
}

// Vector intrinsics.
// These all map directly to z instructions, except that some variants ending
// in "s" have a final "int *" that receives the post-instruction CC value.

// Vector support instructions (chapter 21 of the PoP)
let Attributes = [NoThrow, Const], Features = "vector" in
  def lcbb : SystemZTargetBuiltin<"unsigned int(void const *, int)">;
let Features = "vector" in {
  def vlbb : SystemZTargetBuiltin<"_Vector<16, signed char>(void const *, int)">;
  def vll : SystemZTargetBuiltin<"_Vector<16, signed char>(unsigned int, void const *)">;
  def vstl : SystemZTargetBuiltin<"void(_Vector<16, signed char>, unsigned int, void *)">;
}
let Attributes = [NoThrow, Const], Features = "vector" in {
  def vperm : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vpdi : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)">;
  def vpksh : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>)">;
  def vpkshs : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<8, signed short>, _Vector<8, signed short>, int *)">;
  def vpksf : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>)">;
  def vpksfs : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<4, signed int>, _Vector<4, signed int>, int *)">;
  def vpksg : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vpksgs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)">;
  def vpklsh : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vpklshs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vpklsf : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vpklsfs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vpklsg : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vpklsgs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)">;
  def vuphb : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>)">;
  def vuphh : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>)">;
  def vuphf : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>)">;
  def vuplb : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>)">;
  def vuplhw : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>)">;
  def vuplf : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>)">;
  def vuplhb : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)">;
  def vuplhh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>)">;
  def vuplhf : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)">;
  def vupllb : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)">;
  def vupllh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>)">;
  def vupllf : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>)">;
}

// Vector integer instructions (chapter 22 of the PoP)
let Attributes = [NoThrow, Const], Features = "vector" in {
  def vaq : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t)">;
  def vacq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t, __uint128_t)">;
  def vaccb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vacch : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vaccf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vaccg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vaccq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t)">;
  def vacccq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t, __uint128_t)">;
  def vavgb : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)">;
  def vavgh : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)">;
  def vavgf : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)">;
  def vavgg : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vavglb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vavglh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vavglf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vavglg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vceqbs : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vceqhs : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vceqfs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vceqgs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)">;
  def vchbs : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, int *)">;
  def vchhs : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, int *)">;
  def vchfs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, int *)">;
  def vchgs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, int *)">;
  def vchlbs : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vchlhs : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vchlfs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vchlgs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int *)">;
  def vcksm : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vclzb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)">;
  def vclzh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)">;
  def vclzf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)">;
  def vclzg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)">;
  def vctzb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)">;
  def vctzh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)">;
  def vctzf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)">;
  def vctzg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)">;
  def verimb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def verimh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)">;
  def verimf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)">;
  def verimg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, int)">;
  def verllb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, unsigned char)">;
  def verllh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, unsigned char)">;
  def verllf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, unsigned char)">;
  def verllg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, unsigned char)">;
  def verllvb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def verllvh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def verllvf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def verllvg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vgfmb : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vgfmh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vgfmf : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vgfmg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vgfmab : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)">;
  def vgfmah : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)">;
  def vgfmaf : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)">;
  def vgfmag : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __uint128_t)">;
  def vmahb : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<16, signed char>)">;
  def vmahh : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<8, signed short>)">;
  def vmahf : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<4, signed int>)">;
  def vmalhb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vmalhh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vmalhf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vmaeb : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)">;
  def vmaeh : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)">;
  def vmaef : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)">;
  def vmaleb : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)">;
  def vmaleh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)">;
  def vmalef : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)">;
  def vmaob : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>, _Vector<8, signed short>)">;
  def vmaoh : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>, _Vector<4, signed int>)">;
  def vmaof : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>, _Vector<2, signed long long int>)">;
  def vmalob : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<8, unsigned short>)">;
  def vmaloh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<4, unsigned int>)">;
  def vmalof : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<2, unsigned long long int>)">;
  def vmhb : SystemZTargetBuiltin<"_Vector<16, signed char>(_Vector<16, signed char>, _Vector<16, signed char>)">;
  def vmhh : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<8, signed short>, _Vector<8, signed short>)">;
  def vmhf : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, signed int>, _Vector<4, signed int>)">;
  def vmlhb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vmlhh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vmlhf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vmeb : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)">;
  def vmeh : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)">;
  def vmef : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)">;
  def vmleb : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vmleh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vmlef : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vmob : SystemZTargetBuiltin<"_Vector<8, signed short>(_Vector<16, signed char>, _Vector<16, signed char>)">;
  def vmoh : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<8, signed short>, _Vector<8, signed short>)">;
  def vmof : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<4, signed int>, _Vector<4, signed int>)">;
  def vmlob : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vmloh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vmlof : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vsq : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t)">;
  def vsbcbiq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t, __uint128_t)">;
  def vsbiq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t, __uint128_t)">;
  def vscbib : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vscbih : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vscbif : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vscbig : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vscbiq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t)">;
  def vsl : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vslb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsldb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vsra : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsrab : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsrl : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsrlb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsumb : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vsumh : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vsumgh : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vsumgf : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vsumqf : SystemZTargetBuiltin<"__uint128_t(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vsumqg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vtm : SystemZTargetBuiltin<"int(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
}

// Vector string instructions (chapter 23 of the PoP)
let Attributes = [NoThrow, Const], Features = "vector" in {
  def vfaeb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vfaebs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)">;
  def vfaeh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)">;
  def vfaehs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)">;
  def vfaef : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)">;
  def vfaefs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)">;
  def vfaezb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vfaezbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)">;
  def vfaezh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int)">;
  def vfaezhs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)">;
  def vfaezf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int)">;
  def vfaezfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)">;
  def vfeeb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vfeebs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vfeeh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vfeehs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vfeef : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vfeefs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vfeezb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vfeezbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vfeezh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vfeezhs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vfeezf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vfeezfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vfeneb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vfenebs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vfeneh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vfenehs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vfenef : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vfenefs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vfenezb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vfenezbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vfenezh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>)">;
  def vfenezhs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, int *)">;
  def vfenezf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>)">;
  def vfenezfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, int *)">;
  def vistrb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>)">;
  def vistrbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, int *)">;
  def vistrh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)">;
  def vistrhs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int *)">;
  def vistrf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)">;
  def vistrfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, int *)">;
  def vstrcb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vstrcbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)">;
  def vstrch : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)">;
  def vstrchs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)">;
  def vstrcf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)">;
  def vstrcfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)">;
  def vstrczb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vstrczbs : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int, int *)">;
  def vstrczh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int)">;
  def vstrczhs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<8, unsigned short>, int, int *)">;
  def vstrczf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int)">;
  def vstrczfs : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<4, unsigned int>, int, int *)">;
}

// Vector floating-point instructions (chapter 24 of the PoP)
let Attributes = [NoThrow, Const], Features = "vector" in {
  def vfcedbs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)">;
  def vfchdbs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)">;
  def vfchedbs : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, _Vector<2, double>, int *)">;
  def vfidb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, int, int)">;
  def vflndb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>)">;
  def vflpdb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>)">;
  def vfmadb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
  def vfmsdb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
  def vfsqdb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>)">;
  def vftcidb : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, double>, int, int *)">;
}

// Vector-enhancements facility 1 intrinsics.
let Features = "vector-enhancements-1" in {
  def vlrlr : SystemZTargetBuiltin<"_Vector<16, signed char>(unsigned int, void const *)">;
  def vstrlr : SystemZTargetBuiltin<"void(_Vector<16, signed char>, unsigned int, void *)">;
}
let Attributes = [NoThrow, Const], Features = "vector-enhancements-1" in {
  def vbperm : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<16, unsigned char>, _Vector<16, unsigned char>)">;
  def vmslg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __uint128_t, int)">;
  def vfmaxdb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)">;
  def vfmindb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, int)">;
  def vfnmadb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
  def vfnmsdb : SystemZTargetBuiltin<"_Vector<2, double>(_Vector<2, double>, _Vector<2, double>, _Vector<2, double>)">;
  def vfcesbs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)">;
  def vfchsbs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)">;
  def vfchesbs : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, float>, _Vector<4, float>, int *)">;
  def vfisb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, int, int)">;
  def vfmaxsb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)">;
  def vfminsb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, int)">;
  def vflnsb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>)">;
  def vflpsb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>)">;
  def vfmasb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfmssb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfnmasb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfnmssb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>, _Vector<4, float>, _Vector<4, float>)">;
  def vfsqsb : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<4, float>)">;
  def vftcisb : SystemZTargetBuiltin<"_Vector<4, signed int>(_Vector<4, float>, int, int *)">;
}

// Vector-enhancements facility 2 intrinsics.
let Attributes = [NoThrow, Const], Features = "vector-enhancements-2" in {
  def vsld : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vsrd : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vstrsb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vstrsh : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)">;
  def vstrsf : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)">;
  def vstrszb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int *)">;
  def vstrszh : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>, _Vector<8, unsigned short>, _Vector<16, unsigned char>, int *)">;
  def vstrszf : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<4, unsigned int>, _Vector<4, unsigned int>, _Vector<16, unsigned char>, int *)">;
}

// Helpers to implement vec_revb.
let Attributes = [NoThrow, Const], Features = "vector" in {
  def vlbrh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>)">;
  def vlbrf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<4, unsigned int>)">;
  def vlbrg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>)">;
  def vlbrq : SystemZTargetBuiltin<"__uint128_t(__uint128_t)">;
}

// NNP-assist facility intrinsics.
let Attributes = [NoThrow, Const], Features = "nnp-assist" in {
  def vclfnhs : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<8, unsigned short>, int)">;
  def vclfnls : SystemZTargetBuiltin<"_Vector<4, float>(_Vector<8, unsigned short>, int)">;
  def vcrnfs : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<4, float>, _Vector<4, float>, int)">;
  def vcfn : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)">;
  def vcnf : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<8, unsigned short>, int)">;
}

// Miscellaneous instruction extensions facility 4 intrinsics.
let Attributes = [NoThrow, Const], Features = "miscellaneous-extensions-4" in {
  def bdepg : SystemZTargetBuiltin<"unsigned long int(unsigned long int, unsigned long int)">;
  def bextg : SystemZTargetBuiltin<"unsigned long int(unsigned long int, unsigned long int)">;
}

// Vector-enhancements facility 3 intrinsics.
let Attributes = [NoThrow, Const], Features = "vector-enhancements-3" in {
  def vgemb : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<8, unsigned short>)">;
  def vgemh : SystemZTargetBuiltin<"_Vector<8, unsigned short>(_Vector<16, unsigned char>)">;
  def vgemf : SystemZTargetBuiltin<"_Vector<4, unsigned int>(_Vector<16, unsigned char>)">;
  def vgemg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<16, unsigned char>)">;
  def vgemq : SystemZTargetBuiltin<"__uint128_t(_Vector<16, unsigned char>)">;
  def vuplg : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>)">;
  def vupllg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>)">;
  def vuphg : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>)">;
  def vuplhg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>)">;
  def vavgq : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t)">;
  def vavglq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t)">;
  def veval : SystemZTargetBuiltin<"_Vector<16, unsigned char>(_Vector<16, unsigned char>, _Vector<16, unsigned char>, _Vector<16, unsigned char>, int)">;
  def vmahg : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vmahq : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t, __int128_t)">;
  def vmalhg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vmalhq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t, __uint128_t)">;
  def vmaeg : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>, __int128_t)">;
  def vmaleg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __uint128_t)">;
  def vmaog : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>, __int128_t)">;
  def vmalog : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>, __uint128_t)">;
  def vmhg : SystemZTargetBuiltin<"_Vector<2, signed long long int>(_Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vmhq : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t)">;
  def vmlhg : SystemZTargetBuiltin<"_Vector<2, unsigned long long int>(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vmlhq : SystemZTargetBuiltin<"__uint128_t(__uint128_t, __uint128_t)">;
  def vmeg : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vmleg : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vmog : SystemZTargetBuiltin<"__int128_t(_Vector<2, signed long long int>, _Vector<2, signed long long int>)">;
  def vmlog : SystemZTargetBuiltin<"__uint128_t(_Vector<2, unsigned long long int>, _Vector<2, unsigned long long int>)">;
  def vceqqs : SystemZTargetBuiltin<"__int128_t(__uint128_t, __uint128_t, int *)">;
  def vchqs : SystemZTargetBuiltin<"__int128_t(__int128_t, __int128_t, int *)">;
  def vchlqs : SystemZTargetBuiltin<"__int128_t(__uint128_t, __uint128_t, int *)">;
  def vclzq : SystemZTargetBuiltin<"__uint128_t(__uint128_t)">;
  def vctzq : SystemZTargetBuiltin<"__uint128_t(__uint128_t)">;
}
