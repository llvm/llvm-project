// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=X86

// Under __vectorcall, MSVC returns HVAs and HFAs in vector registers even if
// they are not C++14 aggregates, and even from instance methods. Other calling
// conventions keep the usual restrictions and return such types indirectly.
// See https://github.com/llvm/llvm-project/issues/63417.

#define VC __attribute__((vectorcall))
#define CD __attribute__((cdecl))

typedef float m128 __attribute__((__vector_size__(16), __aligned__(16)));

// Non-aggregate HFA (base class, protected member, user-provided ctor).
struct B2 { float a; };
struct HFA2 : B2 { HFA2(); protected: float b; };

HFA2 VC hfa2_vc(HFA2 *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HFA2 @"?hfa2_vc@@YQ?AUHFA2@@PEAU1@@Z"(ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.HFA2 @"?hfa2_vc@@YQ?AUHFA2@@PAU1@@Z"(ptr inreg noundef %x)

HFA2 CD hfa2_cd(HFA2 *x) { return *x; }
// X64: define dso_local void @"?hfa2_cd@@YA?AUHFA2@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HFA2) align 4 %agg.result, ptr noundef %x)
// X86: define dso_local void @"?hfa2_cd@@YA?AUHFA2@@PAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HFA2) align 4 %agg.result, ptr noundef %x)

// Non-aggregate HFA of doubles with only a user-provided constructor.
struct HFAD { HFAD(int); double a, b, c; };

HFAD VC hfad_vc(HFAD *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HFAD @"?hfad_vc@@YQ?AUHFAD@@PEAU1@@Z"(ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.HFAD @"?hfad_vc@@YQ?AUHFAD@@PAU1@@Z"(ptr inreg noundef %x)

HFAD CD hfad_cd(HFAD *x) { return *x; }
// X64: define dso_local void @"?hfad_cd@@YA?AUHFAD@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HFAD) align 8 %agg.result, ptr noundef %x)
// X86: define dso_local void @"?hfad_cd@@YA?AUHFAD@@PAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HFAD) align 8 %agg.result, ptr noundef %x)

// Non-aggregate HVA.
struct BV { m128 v1; };
struct HVA2 : BV { HVA2(double); protected: m128 v2; };

HVA2 VC hva2_vc(HVA2 *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HVA2 @"?hva2_vc@@YQ?AUHVA2@@PEAU1@@Z"(ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.HVA2 @"?hva2_vc@@YQ?AUHVA2@@PAU1@@Z"(ptr inreg noundef %x)

HVA2 CD hva2_cd(HVA2 *x) { return *x; }
// X64: define dso_local void @"?hva2_cd@@YA?AUHVA2@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HVA2) align 16 %agg.result, ptr noundef %x)
// X86: define dso_local void @"?hva2_cd@@YA?AUHVA2@@PAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HVA2) align 16 %agg.result, ptr noundef %x)

// Non-trivial destructor. On x64, small classes with a non-trivial destructor
// can still be passed in registers, and MSVC returns them in XMM registers. On
// x86 they are returned indirectly.
struct HFADtor { ~HFADtor(); float a, b; };

HFADtor VC hfadtor_vc(HFADtor *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HFADtor @"?hfadtor_vc@@YQ?AUHFADtor@@PEAU1@@Z"(ptr noundef %x)
// X86: define dso_local x86_vectorcallcc void @"?hfadtor_vc@@YQ?AUHFADtor@@PAU1@@Z"(ptr dead_on_unwind inreg noalias writable sret(%struct.HFADtor) align 4 %agg.result, ptr inreg noundef %x)

// Too many elements to be an HVA: still returned indirectly.
struct HFA5 { HFA5(); float a, b, c, d, e; };

HFA5 VC hfa5_vc(HFA5 *x) { return *x; }
// X64: define dso_local x86_vectorcallcc void @"?hfa5_vc@@YQ?AUHFA5@@PEAU1@@Z"(ptr dead_on_unwind noalias writable sret(%struct.HFA5) align 4 %agg.result, ptr noundef %x)
// X86: define dso_local x86_vectorcallcc void @"?hfa5_vc@@YQ?AUHFA5@@PAU1@@Z"(ptr dead_on_unwind inreg noalias writable sret(%struct.HFA5) align 4 %agg.result, ptr inreg noundef %x)

// Instance methods.
struct Agg { float a, b; };
struct S {
  Agg VC agg_vc(Agg *x);
  HFA2 VC hfa2_vc(HFA2 *x);
  Agg agg_tc(Agg *x);
  static HFA2 VC static_hfa2_vc(HFA2 *x);
};

Agg VC S::agg_vc(Agg *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.Agg @"?agg_vc@S@@QEAQ?AUAgg@@PEAU2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.Agg @"?agg_vc@S@@QAQ?AUAgg@@PAU2@@Z"(ptr inreg noundef nonnull align 1 dereferenceable(1) %this, ptr inreg noundef %x)

HFA2 VC S::hfa2_vc(HFA2 *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HFA2 @"?hfa2_vc@S@@QEAQ?AUHFA2@@PEAU2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.HFA2 @"?hfa2_vc@S@@QAQ?AUHFA2@@PAU2@@Z"(ptr inreg noundef nonnull align 1 dereferenceable(1) %this, ptr inreg noundef %x)

Agg S::agg_tc(Agg *x) { return *x; }
// X64: define dso_local void @"?agg_tc@S@@QEAA?AUAgg@@PEAU2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr dead_on_unwind noalias writable sret(%struct.Agg) align 4 %agg.result, ptr noundef %x)
// X86: define dso_local x86_thiscallcc void @"?agg_tc@S@@QAE?AUAgg@@PAU2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr dead_on_unwind noalias writable sret(%struct.Agg) align 4 %agg.result, ptr noundef %x)

HFA2 VC S::static_hfa2_vc(HFA2 *x) { return *x; }
// X64: define dso_local x86_vectorcallcc %struct.HFA2 @"?static_hfa2_vc@S@@SQ?AUHFA2@@PEAU2@@Z"(ptr noundef %x)
// X86: define dso_local x86_vectorcallcc %struct.HFA2 @"?static_hfa2_vc@S@@SQ?AUHFA2@@PAU2@@Z"(ptr inreg noundef %x)
