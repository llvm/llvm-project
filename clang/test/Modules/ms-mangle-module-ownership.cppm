// Windows ABI module-ownership mangling, for
// https://github.com/llvm/llvm-project/issues/174064.
//
// MSVC tags externally-visible module-owned entities with a
// "::<!module-name>" suffix, and a module-owned record referenced as an
// embedded type (e.g. in RTTI, or a cross-module parameter) with a
// "$$_A<module-name>" scope component. Expected encodings below are taken
// from real cl.exe output.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/a.pcm -c %t/a.cppm -o %t/a.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/partition.pcm -c %t/partition.cppm -o %t/partition.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/long.pcm -c %t/long.cppm -o %t/long.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/hashed.pcm -c %t/hashed.cppm -o %t/hashed.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/a.cppm -o - | FileCheck %t/a.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/partition.cppm -o - | FileCheck %t/partition.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/nonmodule.cpp -o - | FileCheck %t/nonmodule.cpp
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/long.cppm -o - | FileCheck %t/long.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/hashed.cppm -o - | FileCheck %t/hashed.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/gmf.cppm -o - | FileCheck %t/gmf.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modb.pcm -c %t/modb.cppm -o %t/modb.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=b=%t/modb.pcm -S -emit-llvm %t/crossmod.cppm -o - | FileCheck %t/crossmod.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=b=%t/modb.pcm -S -emit-llvm %t/crossmod_val.cppm -o - | FileCheck %t/crossmod_val.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=b=%t/modb.pcm -S -emit-llvm %t/tmplcross.cppm -o - | FileCheck %t/tmplcross.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=b=%t/modb.pcm -S -emit-llvm %t/nonmodule_cross.cpp -o - | FileCheck %t/nonmodule_cross.cpp
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modc.pcm -c %t/modc.cppm -o %t/modc.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=c=%t/modc.pcm -S -emit-llvm %t/functmplcross.cppm -o - | FileCheck %t/functmplcross.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modd.pcm -c %t/modd.cppm -o %t/modd.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=d=%t/modd.pcm -fmodule-output=%t/mode.pcm -c %t/mode.cppm -o %t/mode.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=d=%t/modd.pcm -fmodule-file=e=%t/mode.pcm -S -emit-llvm %t/reexport.cppm -o - | FileCheck %t/reexport.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modg.pcm -c %t/modg.cppm -o %t/modg.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=g=%t/modg.pcm -S -emit-llvm %t/crossenum.cppm -o - | FileCheck %t/crossenum.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -S -emit-llvm %t/dotmod.cppm -o - | FileCheck %t/dotmod.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modm.pcm -c %t/gmfreexport.cppm -o %t/gmfreexport.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=m=%t/modm.pcm -S -emit-llvm %t/usegmfreexport.cppm -o - | FileCheck %t/usegmfreexport.cppm
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-output=%t/modparta.pcm -c %t/modparta.cppm -o %t/modparta.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=mod2:parta=%t/modparta.pcm -fmodule-output=%t/modpartb.pcm -c %t/modpartb.cppm -o %t/modpartb.o
// RUN: %clang -std=c++20 --target=x86_64-windows-msvc -fmodule-file=mod2:parta=%t/modparta.pcm -fmodule-file=mod2:partb=%t/modpartb.pcm -S -emit-llvm %t/modaggregate.cppm -o - | FileCheck %t/modaggregate.cppm

//--- a.cppm
export module a;

export int func() { return 43; }

int nonExportedFunc() { return 2; }
static int internalFunc() { return 3; }

export int gVar = 42;

export struct Foo {
    static int sVal;
    int x;
    virtual int f() { return 1; }
    virtual ~Foo() {}
};
int Foo::sVal = 1;

export int takesFoo(Foo p) { return p.x; }
export Foo makesFoo() { return Foo(); }

// A class template instantiation is owned by the template's own module, so
// TBox<int> (same module "a" as its use here) mangles like any other
// same-module type -- unmarked when embedded, but still tagged in RTTI.
export template <typename T>
struct TBox {
    T val;
    T get() { return val; }
    virtual ~TBox() {}
};
export int useTBox() {
    TBox<int> b;
    b.val = 1;
    return b.get();
}
export TBox<int> makesTBox() { return TBox<int>(); }

// Referencing internalFunc keeps it from being discarded as unused, without
// changing its (internal) linkage.
export int useInternal() { return internalFunc(); }

// Module linkage (plain or non-exported) gets the suffix; true internal
// linkage (`static`) does not. A struct passed by value only tags the
// enclosing function's own name, not the embedded parameter/return type
// (same module). Base and scalar-deleting dtors get the suffix;
// vector-deleting does not. RTTI's type descriptor gets "$$_A<module>@@"
// instead of the plain suffix.
//
// CHECK-DAG lines are kept in one contiguous block (FileCheck's DAG search
// floor) with CHECK-NOTs moved after, since a CHECK-NOT between DAG groups
// would skip matches appearing before it in the IR.
// CHECK-DAG: define dso_local {{.*}} @"?func@@YAHXZ::<!a>"(
// CHECK-DAG: define dso_local {{.*}} @"?nonExportedFunc@@YAHXZ::<!a>"(
// CHECK-DAG: define internal {{.*}} @"?internalFunc@@YAHXZ"(
// CHECK-DAG: @"?gVar@@3HA::<!a>" =
// CHECK-DAG: @"?sVal@Foo@@2HA::<!a>" =
// CHECK-DAG: define {{.*}} @"?takesFoo@@YAHUFoo@@@Z::<!a>"(
// CHECK-DAG: define {{.*}} @"?makesFoo@@YA?AUFoo@@XZ::<!a>"(
// CHECK-DAG: define {{.*}} @"??1Foo@@UEAA@XZ::<!a>"(
// CHECK-DAG: define {{.*}} @"??_GFoo@@UEAAPEAXI@Z::<!a>"(
// CHECK-DAG: @"??_R0?AUFoo@$$_Aa@@@@8" =
// CHECK-DAG: define {{.*}} @"?get@?$TBox@H@@QEAAHXZ::<!a>"(
// CHECK-DAG: define {{.*}} @"??1?$TBox@H@@UEAA@XZ::<!a>"(
// CHECK-DAG: define {{.*}} @"?makesTBox@@YA?AU?$TBox@H@@XZ::<!a>"(
// CHECK-DAG: @"??_R0?AU?$TBox@H@$$_Aa@@@@8" =
//
// The vftable and other RTTI symbols (Complete Object Locator, Class
// Hierarchy Descriptor, Base Class Array, Base Class Descriptor) get no
// module suffix at all -- only the type descriptor does.
// CHECK-DAG: @"??_7Foo@@6B@" =
// CHECK-DAG: @"??_R4Foo@@6B@" =
// CHECK-DAG: @"??_R3Foo@@8" =
// CHECK-DAG: @"??_R2Foo@@8" =
// CHECK-DAG: @"??_R1A@?0A@EA@Foo@@8" =
// CHECK-NOT: internalFunc{{.*}}::<!
// CHECK-NOT: "??_EFoo@@UEAAPEAXI@Z::<!

//--- partition.cppm
export module a:b;

export int partFunc() { return 1; }

export struct Foo {
    virtual int f() { return 1; }
    virtual ~Foo() {}
};
Foo *make() { return new Foo(); }

// A module partition's suffix drops the partition part entirely: only the
// primary module name is used, both for the plain suffix and for RTTI.
// CHECK-DAG: define {{.*}} @"?partFunc@@YAHXZ::<!a>"(
// CHECK-DAG: @"??_R0?AUFoo@$$_Aa@@@@8" =

//--- nonmodule.cpp
struct Foo {
    virtual int f() { return 1; }
    virtual ~Foo() {}
};
Foo *make() { return new Foo(); }

// No module ownership at all -> ordinary, unmodified RTTI type descriptor.
// CHECK: @"??_R0?AUFoo@@@8" =
// CHECK-NOT: $$_A

//--- long.cppm
export module abcdefgh;

export struct Foo {
    virtual int f() { return 1; }
    virtual ~Foo() {}
};
Foo *make() { return new Foo(); }

// The RTTI marker uses the module name raw, with no length prefix.
// CHECK: @"??_R0?AUFoo@$$_Aabcdefgh@@@@8" =

//--- hashed.cppm
module;
// Enough repeated "int" parameters (each mangles as a single 'H') to push
// wideFunc's mangled name past msvc_hashing_ostream's 4096-char threshold;
// doubling macros keep the source short while generating thousands of
// tokens in the parameter list.
#define I1 int
#define I2 I1, I1
#define I4 I2, I2
#define I8 I4, I4
#define I16 I8, I8
#define I32 I16, I16
#define I64 I32, I32
#define I128 I64, I64
#define I256 I128, I128
#define I512 I256, I256
#define I1024 I512, I512
#define I2048 I1024, I1024
#define I4096 I2048, I2048
#define I4608 I4096, I512
export module a;

export int wideFunc(I4608) { return 0; }

// Past the hashing threshold the whole mangled name is replaced with
// "??@<md5>@", but the module suffix is still appended after that hash,
// not folded into it.
// CHECK: define {{.*}} @"??@{{[0-9a-f]+}}@::<!a>"(

//--- gmf_common.h
int gmfFunc() { return 1; }
struct GmfFoo {
    virtual int f() { return 1; }
    virtual ~GmfFoo() {}
};

//--- gmf.cppm
module;
#include "gmf_common.h"
export module a;

export int useGmf() { return gmfFunc(); }
export GmfFoo *makeGmfFoo() { return new GmfFoo(); }

// Global-module-fragment entities (reached via #include before the module
// declaration) are not module-owned -- only the module-purview functions
// referencing them get the suffix.
// CHECK-DAG: define {{.*}} @"?useGmf@@YAHXZ::<!a>"(
// CHECK-DAG: define {{.*}} @"?makeGmfFoo@@YAPEAUGmfFoo@@XZ::<!a>"(
// CHECK-DAG: define {{.*}} @"?gmfFunc@@YAHXZ"(
// CHECK-DAG: define {{.*}} @"?f@GmfFoo@@UEAAHXZ"(
// CHECK-DAG: define {{.*}} @"??1GmfFoo@@UEAA@XZ"(
// CHECK-DAG: @"??_R0?AUGmfFoo@@@8" =
// CHECK-NOT: gmfFunc{{.*}}::<!
// CHECK-NOT: GmfFoo{{.*}}::<!
// CHECK-NOT: $$_A

//--- modb.cppm
export module b;

export struct BFoo {
    virtual int f() { return 1; }
    virtual ~BFoo() {}
};

export template <typename T>
struct Box {
    T val;
    T get() { return val; }
    virtual ~Box() {}
};

//--- crossmod.cppm
export module a;
import b;

export BFoo *takesB(BFoo *p) { return p; }

// A cross-module record gets the "$$_A<module>" marker even in an ordinary
// function signature, not just RTTI, and it shares the ordinary name
// back-reference table: BFoo* appears as both return type and parameter,
// so the second occurrence is a back-reference digit ("$$_A2") rather than
// the name again -- and needs one fewer trailing '@' than a fresh name.
// CHECK: define {{.*}} @"?takesB@@YAPEAUBFoo@$$_Ab@@@PEAU1$$_A2@@@Z::<!a>"(

//--- crossmod_val.cppm
export module a;
import b;

export int takesB2(BFoo p, BFoo q) { return 0; }

// The same cross-module type by value twice: MS mangling's separate
// whole-argument-type substitution table recognizes the repeat and skips
// re-mangling it (back-reference "0"), never touching the module-name
// back-reference table at all.
// CHECK: define {{.*}} @"?takesB2@@YAHUBFoo@$$_Ab@@@0@Z::<!a>"(

//--- tmplcross.cppm
export module a;
import b;

export int useBox() {
    Box<int> b;
    b.val = 1;
    return b.get();
}
export Box<int> makeBox() { return Box<int>(); }

// Box<int> here is owned by "b" (the template's module), not "a" -- its
// member functions get "::<!b>", and referencing it from makeBox (owned by
// "a") now crosses modules, so it also gets the "$$_Ab" embedded marker
// and an "b"-tagged RTTI descriptor.
// CHECK-DAG: define {{.*}} @"?useBox@@YAHXZ::<!a>"(
// CHECK-DAG: define {{.*}} @"?get@?$Box@H@@QEAAHXZ::<!b>"(
// CHECK-DAG: define {{.*}} @"??1?$Box@H@@UEAA@XZ::<!b>"(
// CHECK-DAG: define {{.*}} @"?makeBox@@YA?AU?$Box@H@$$_Ab@@@XZ::<!a>"(
// CHECK-DAG: @"??_R0?AU?$Box@H@$$_Ab@@@@8" =

//--- nonmodule_cross.cpp
import b;

BFoo *takesBFromPlainTU(BFoo *p) { return p; }

// A plain, non-module translation unit (no "export module" at all) that
// imports a module and references its type: takesBFromPlainTU itself gets
// no "::<!module>" suffix (it isn't module-owned), but BFoo -- which is --
// still gets the "$$_Ab" embedded tag. MangleContextModule ends up null
// here for the same reason it's always null for RTTI: the enclosing
// entity just has no module of its own to compare against, so any
// module-owned type reference is unconditionally tagged.
// CHECK: define {{.*}} @"?takesBFromPlainTU@@YAPEAUBFoo@$$_Ab@@@PEAU1$$_A2@@@Z"(
// CHECK-NOT: ::<!

//--- modc.cppm
export module c;

export template <typename T>
T identity(T v) { return v; }

//--- functmplcross.cppm
export module a2;
import c;

export int useIdentity() { return identity(5); }

// Function templates follow the same rule: identity<int> is owned by "c"
// (the template's module), the caller by "a2".
// CHECK-DAG: define {{.*}} @"?useIdentity@@YAHXZ::<!a2>"(
// CHECK-DAG: define {{.*}} @"??$identity@H@@YAHH@Z::<!c>"(

//--- modd.cppm
export module d;

export struct DFoo {
    virtual int f() { return 1; }
    virtual ~DFoo() {}
};
export int dFunc() { return 1; }

//--- mode.cppm
export module e;
export import d;

//--- reexport.cppm
export module f;
import e;

export DFoo *useDFoo(DFoo *p) { return p; }
export int useDFunc() { return dFunc(); }

// "f" imports "d" only indirectly, through "e"'s "export import d;". A
// re-export never changes an entity's true owning module: DFoo and dFunc
// still mangle as owned by "d", not "e".
// CHECK-DAG: define {{.*}} @"?useDFoo@@YAPEAUDFoo@$$_Ad@@@PEAU1$$_A2@@@Z::<!f>"(
// CHECK-DAG: define {{.*}} @"?useDFunc@@YAHXZ::<!f>"(
// CHECK-DAG: declare {{.*}} @"?dFunc@@YAHXZ::<!d>"(

//--- modg.cppm
export module g;

export enum class Color { Red, Green, Blue };

//--- crossenum.cppm
export module h;
import g;

export Color useColor(Color c) { return c; }

// mangleType(const TagDecl*) is the shared choke point for both records
// and enums, so a cross-module enum gets the same embedded marker a struct
// would (just with the "W4" tag-kind letter instead of "U").
// CHECK: define {{.*}} @"?useColor@@YA?AW4Color@$$_Ag@@@W41$$_A2@@@Z::<!h>"(

//--- dotmod.cppm
export module FOO.BAR;

export int func() { return 1; }

// A dot in a module name has no special meaning to MSVC's mangler: unlike
// Itanium (which mangles "FOO.BAR" as two separately-substitutable
// <module-subname> components), MSVC emits the primary module name as one
// literal string.
// CHECK: define {{.*}} @"?func@@YAHXZ::<!FOO.BAR>"(

//--- gmfreexport.cppm
module;
int gmfFunc() { return 1; }
export module m;

export using ::gmfFunc;

//--- usegmfreexport.cppm
export module n;
import m;

export int useIt() { return gmfFunc(); }

// "export using" re-exporting a GMF entity does not retroactively give it
// module ownership -- gmfFunc still mangles with no suffix, same as an
// unexported GMF reference.
// CHECK-DAG: define {{.*}} @"?useIt@@YAHXZ::<!n>"(
// CHECK-DAG: declare {{.*}} @"?gmfFunc@@YAHXZ"(

//--- modparta.cppm
export module mod2:parta;

export int a2 = 43;
export int foo2() { return 3 + a2; }

//--- modpartb.cppm
module mod2:partb;

int b2 = 43;
int bar2() { return 43 + b2; }

//--- modaggregate.cppm
export module mod2;
import :parta;
import :partb;

export int use2() { return foo2() + bar2() + a2 + b2; }

// A primary module interface aggregating an exported partition (":parta")
// and an internal one (":partb") across a real multi-file build: both
// still mangle with the primary module name only, export status making no
// difference.
// CHECK-DAG: declare {{.*}} @"?foo2@@YAHXZ::<!mod2>"(
// CHECK-DAG: declare {{.*}} @"?bar2@@YAHXZ::<!mod2>"(
// CHECK-DAG: define {{.*}} @"?use2@@YAHXZ::<!mod2>"(
// CHECK-DAG: @"?a2@@3HA::<!mod2>" =
// CHECK-DAG: @"?b2@@3HA::<!mod2>" =
