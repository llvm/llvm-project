// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O1 -mconstructor-aliases -disable-llvm-optzns -o - %s -w -fms-compatibility-version=19.00 | FileCheck --check-prefix=MSC --check-prefix=M32 -check-prefix=MSVC2015 -check-prefix=M32MSVC2015 %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O1 -mconstructor-aliases -disable-llvm-optzns -o - %s -w -fms-compatibility-version=18.00 | FileCheck --check-prefix=MSC --check-prefix=M32 -check-prefix=MSVC2013 -check-prefix=M32MSVC2013 %s

// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s -w -fms-compatibility-version=19.00 | FileCheck --check-prefix=MSC --check-prefix=M64 -check-prefix=MSVC2015 -check-prefix=M64MSVC2015 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s -w -fms-compatibility-version=18.00 | FileCheck --check-prefix=MSC --check-prefix=M64 -check-prefix=MSVC2013 -check-prefix=M64MSVC2013 %s

// RUN: %clang_cc1 -triple i686-windows-gnu    -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s -w | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s -w | FileCheck --check-prefix=GNU --check-prefix=G64 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Exported {};
struct ExplicitDecl_Exported {};
struct ExplicitInst_Exported {};
struct ExplicitSpec_Exported {};
struct ExplicitSpec_Def_Exported {};
struct ExplicitSpec_InlineDef_Exported {};
struct ExplicitSpec_NotExported {};
struct External { int v; };

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEVAR(var) int UNIQ(use)() { return var; }
#define USE(func) void UNIQ(use)() { func(); }
#define USEMEMFUNC(class, func) void (class::*UNIQ(use)())() { return &class::func; }
#define INSTVAR(var) template int var;
#define INST(func) template void func();

// The vftable for struct W is comdat largest because we have RTTI.
// M32-DAG: $"\01??_7W@@6B@" = comdat largest


//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Declarations are not exported.
// MSC-NOT: @"\01?ExternGlobalDecl@@3HA"
// GNU-NOT: @ExternGlobalDecl
__declspec(dllexport) extern int ExternGlobalDecl;

// dllexport implies a definition.
// MSC-DAG: @"\01?GlobalDef@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalDef            = dllexport global i32 0, align 4
__declspec(dllexport) int GlobalDef;

// Export definition.
// MSC-DAG: @"\01?GlobalInit1@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalInit1            = dllexport global i32 1, align 4
__declspec(dllexport) int GlobalInit1 = 1;

// MSC-DAG: @"\01?GlobalInit2@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalInit2            = dllexport global i32 1, align 4
int __declspec(dllexport) GlobalInit2 = 1;

// Declare, then export definition.
// MSC-DAG: @"\01?GlobalDeclInit@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @GlobalDeclInit            = dllexport global i32 1, align 4
__declspec(dllexport) extern int GlobalDeclInit;
int GlobalDeclInit = 1;

// Redeclarations
// MSC-DAG: @"\01?GlobalRedecl1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalRedecl1            = dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl1;
__declspec(dllexport)        int GlobalRedecl1;

// MSC-DAG: @"\01?GlobalRedecl2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @GlobalRedecl2            = dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl2;
                             int GlobalRedecl2;

// MSC-DAG: @"\01?ExternalGlobal@ns@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_ZN2ns14ExternalGlobalE      = dllexport global i32 0, align 4
namespace ns { __declspec(dllexport) int ExternalGlobal; }

// MSC-DAG: @"\01?ExternalAutoTypeGlobal@@3UExternal@@A" = dllexport global %struct.External zeroinitializer, align 4
// GNU-DAG: @ExternalAutoTypeGlobal                      = dllexport global %struct.External zeroinitializer, align 4
__declspec(dllexport) auto ExternalAutoTypeGlobal = External();

int f();
// MSC-DAG: @"\01?x@?1??nonInlineStaticLocalsFunc@@YAHXZ@4HA" = internal {{(unnamed_addr )*}}global i32 0
// MSC-DAG: @"\01?$S1@?1??nonInlineStaticLocalsFunc@@YAHXZ@4IA" = internal {{(unnamed_addr )*}}global i32 0
int __declspec(dllexport) nonInlineStaticLocalsFunc() {
  static int x = f();
  return x++;
};

// MSC-DAG: @"\01?x@?1??inlineStaticLocalsFunc@@YAHXZ@4HA" = weak_odr dllexport global i32 0, comdat
// MSC-DAG: @"\01??_B?1??inlineStaticLocalsFunc@@YAHXZ@51" = weak_odr dllexport global i32 0, comdat
// Note: MinGW doesn't seem to export the static local here.
inline int __declspec(dllexport) inlineStaticLocalsFunc() {
  static int x = f();
  return x++;
}



//===----------------------------------------------------------------------===//
// Variable templates
//===----------------------------------------------------------------------===//

// Declarations are not exported.

// dllexport implies a definition.
// MSC-NOT: @"\01??$VarTmplDef@UExplicitInst_Exported@@@@3HA"
// GNU-NOT: @_Z10VarTmplDefI21ExplicitInst_ExportedE
template<typename T> __declspec(dllexport) int VarTmplDef;
INSTVAR(VarTmplDef<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplImplicitDef@UImplicitInst_Exported@@@@3HA" = external dllexport global
// GNU-DAG: @_Z18VarTmplImplicitDefI21ImplicitInst_ExportedE          = external dllexport global
template<typename T> __declspec(dllexport) int VarTmplImplicitDef;
USEVAR(VarTmplImplicitDef<ImplicitInst_Exported>)

// Export definition.
// MSC-DAG: @"\01??$VarTmplInit1@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z12VarTmplInit1I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template<typename T> __declspec(dllexport) int VarTmplInit1 = 1;
INSTVAR(VarTmplInit1<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplInit2@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z12VarTmplInit2I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template<typename T> int __declspec(dllexport) VarTmplInit2 = 1;
INSTVAR(VarTmplInit2<ExplicitInst_Exported>)

// Declare, then export definition.
// MSC-DAG: @"\01??$VarTmplDeclInit@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z15VarTmplDeclInitI21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template<typename T> __declspec(dllexport) extern int VarTmplDeclInit;
template<typename T>                              int VarTmplDeclInit = 1;
INSTVAR(VarTmplDeclInit<ExplicitInst_Exported>)

// Redeclarations
// MSC-DAG: @"\01??$VarTmplRedecl1@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z14VarTmplRedecl1I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template<typename T> __declspec(dllexport) extern int VarTmplRedecl1;
template<typename T> __declspec(dllexport)        int VarTmplRedecl1 = 1;
INSTVAR(VarTmplRedecl1<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$VarTmplRedecl2@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z14VarTmplRedecl2I21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template<typename T> __declspec(dllexport) extern int VarTmplRedecl2;
template<typename T>                              int VarTmplRedecl2 = 1;
INSTVAR(VarTmplRedecl2<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$ExternalVarTmpl@UExplicitInst_Exported@@@ns@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_ZN2ns15ExternalVarTmplI21ExplicitInst_ExportedEE        = weak_odr dllexport global i32 1, comdat, align 4
namespace ns { template<typename T> __declspec(dllexport) int ExternalVarTmpl = 1; }
INSTVAR(ns::ExternalVarTmpl<ExplicitInst_Exported>)

// MSC-DAG: @"\01??$ExternalAutoTypeVarTmpl@UExplicitInst_Exported@@@@3UExternal@@A" = weak_odr dllexport global %struct.External zeroinitializer, comdat, align 4
// GNU-DAG: @_Z23ExternalAutoTypeVarTmplI21ExplicitInst_ExportedE                    = weak_odr dllexport global %struct.External zeroinitializer, comdat, align 4
template<typename T> __declspec(dllexport) auto ExternalAutoTypeVarTmpl = External();
template External ExternalAutoTypeVarTmpl<ExplicitInst_Exported>;


template<typename T> int VarTmpl = 1;
template<typename T> __declspec(dllexport) int ExportedVarTmpl = 1;

// Export implicit instantiation of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ImplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
USEVAR(ExportedVarTmpl<ImplicitInst_Exported>)

// Export explicit instantiation declaration of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitDecl_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
extern template int ExportedVarTmpl<ExplicitDecl_Exported>;
       template int ExportedVarTmpl<ExplicitDecl_Exported>;

// Export explicit instantiation definition of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UImplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitInst_ExportedE          = weak_odr dllexport global i32 1, comdat, align 4
template __declspec(dllexport) int ExportedVarTmpl<ExplicitInst_Exported>;

// Export specialization of an exported variable template.
// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_Exported@@@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_Z15ExportedVarTmplI21ExplicitSpec_ExportedE          = dllexport global i32 0, align 4
template<> __declspec(dllexport) int ExportedVarTmpl<ExplicitSpec_Exported>;

// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_Def_Exported@@@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @_Z15ExportedVarTmplI25ExplicitSpec_Def_ExportedE          = dllexport global i32 1, align 4
template<> __declspec(dllexport) int ExportedVarTmpl<ExplicitSpec_Def_Exported> = 1;

// Not exporting specialization of an exported variable template without
// explicit dllexport.
// MSC-DAG: @"\01??$ExportedVarTmpl@UExplicitSpec_NotExported@@@@3HA" = global i32 0, align 4
// GNU-DAG: @_Z15ExportedVarTmplI24ExplicitSpec_NotExportedE          = global i32 0, align 4
template<> int ExportedVarTmpl<ExplicitSpec_NotExported>;


// Export explicit instantiation declaration of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitDecl_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitDecl_ExportedE           = weak_odr dllexport global i32 1, comdat, align 4
extern template __declspec(dllexport) int VarTmpl<ExplicitDecl_Exported>;
       template __declspec(dllexport) int VarTmpl<ExplicitDecl_Exported>;

// Export explicit instantiation definition of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitInst_Exported@@@@3HA" = weak_odr dllexport global i32 1, comdat, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitInst_ExportedE           = weak_odr dllexport global i32 1, comdat, align 4
template __declspec(dllexport) int VarTmpl<ExplicitInst_Exported>;

// Export specialization of a non-exported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Exported@@@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @_Z7VarTmplI21ExplicitSpec_ExportedE           = dllexport global i32 0, align 4
template<> __declspec(dllexport) int VarTmpl<ExplicitSpec_Exported>;

// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Def_Exported@@@@3HA" = dllexport global i32 1, align 4
// GNU-DAG: @_Z7VarTmplI25ExplicitSpec_Def_ExportedE           = dllexport global i32 1, align 4
template<> __declspec(dllexport) int VarTmpl<ExplicitSpec_Def_Exported> = 1;



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Declarations are not exported.

// Export function definition.
// MSC-DAG: define dllexport void @"\01?def@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z3defv()
__declspec(dllexport) void def() {}

// extern "C"
// MSC-DAG: define dllexport void @externC()
// GNU-DAG: define dllexport void @externC()
extern "C" __declspec(dllexport) void externC() {}

// Export inline function.
// MSC-DAG: define weak_odr dllexport void @"\01?inlineFunc@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z10inlineFuncv()
__declspec(dllexport) inline void inlineFunc() {}

// MSC-DAG: define weak_odr dllexport void @"\01?inlineDecl@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z10inlineDeclv()
__declspec(dllexport) inline void inlineDecl();
                             void inlineDecl() {}

// MSC-DAG: define weak_odr dllexport void @"\01?inlineDef@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z9inlineDefv()
__declspec(dllexport) void inlineDef();
               inline void inlineDef() {}

// Redeclarations
// MSC-DAG: define dllexport void @"\01?redecl1@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7redecl1v()
__declspec(dllexport) void redecl1();
__declspec(dllexport) void redecl1() {}

// MSC-DAG: define dllexport void @"\01?redecl2@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7redecl2v()
__declspec(dllexport) void redecl2();
                      void redecl2() {}

// Friend functions
// MSC-DAG: define dllexport void @"\01?friend1@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7friend1v()
// MSC-DAG: define dllexport void @"\01?friend2@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z7friend2v()
struct FuncFriend {
  friend __declspec(dllexport) void friend1();
  friend __declspec(dllexport) void friend2();
};
__declspec(dllexport) void friend1() {}
                      void friend2() {}

// Implicit declarations can be redeclared with dllexport.
// MSC-DAG: define dllexport noalias i8* @"\01??2@{{YAPAXI|YAPEAX_K}}@Z"(
// GNU-DAG: define dllexport noalias i8* @_Znw{{[yj]}}(
void* alloc(__SIZE_TYPE__ n);
__declspec(dllexport) void* operator new(__SIZE_TYPE__ n) { return alloc(n); }

// MSC-DAG: define dllexport void @"\01?externalFunc@ns@@YAXXZ"()
// GNU-DAG: define dllexport void @_ZN2ns12externalFuncEv()
namespace ns { __declspec(dllexport) void externalFunc() {} }



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Export function template definition.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplDef@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z11funcTmplDefI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplDef() {}
INST(funcTmplDef<ExplicitInst_Exported>)

// Export inline function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmpl1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15inlineFuncTmpl1I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) inline void inlineFuncTmpl1() {}
INST(inlineFuncTmpl1<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmpl2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15inlineFuncTmpl2I21ExplicitInst_ExportedEvv()
template<typename T> inline void __attribute__((dllexport)) inlineFuncTmpl2() {}
INST(inlineFuncTmpl2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmplDecl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z18inlineFuncTmplDeclI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) inline void inlineFuncTmplDecl();
template<typename T>                              void inlineFuncTmplDecl() {}
INST(inlineFuncTmplDecl<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$inlineFuncTmplDef@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z17inlineFuncTmplDefI21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void inlineFuncTmplDef();
template<typename T>                inline void inlineFuncTmplDef() {}
INST(inlineFuncTmplDef<ExplicitInst_Exported>)


// Redeclarations
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl1I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl1();
template<typename T> __declspec(dllexport) void funcTmplRedecl1() {}
INST(funcTmplRedecl1<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl2I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl2();
template<typename T>                       void funcTmplRedecl2() {}
INST(funcTmplRedecl2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplRedecl3@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplRedecl3I21ExplicitInst_ExportedEvv()
template<typename T> __declspec(dllexport) void funcTmplRedecl3();
template<typename T>                       void funcTmplRedecl3() {}
INST(funcTmplRedecl3<ExplicitInst_Exported>)


// Function template friends
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplFriend1@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplFriend1I21ExplicitInst_ExportedEvv()
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmplFriend2@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z15funcTmplFriend2I21ExplicitInst_ExportedEvv()
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllexport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllexport) void funcTmplFriend2();
};
template<typename T> __declspec(dllexport) void funcTmplFriend1() {}
template<typename T>                       void funcTmplFriend2() {}
INST(funcTmplFriend1<ExplicitInst_Exported>)
INST(funcTmplFriend2<ExplicitInst_Exported>)

// MSC-DAG: define weak_odr dllexport void @"\01??$externalFuncTmpl@UExplicitInst_Exported@@@ns@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_ZN2ns16externalFuncTmplI21ExplicitInst_ExportedEEvv()
namespace ns { template<typename T> __declspec(dllexport) void externalFuncTmpl() {} }
INST(ns::externalFuncTmpl<ExplicitInst_Exported>)


template<typename T> void funcTmpl() {}
template<typename T> __declspec(dllexport) void exportedFuncTmpl() {}

// Export implicit instantiation of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UImplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ImplicitInst_ExportedEvv()
USE(exportedFuncTmpl<ImplicitInst_Exported>)

// Export explicit instantiation declaration of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitDecl_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ExplicitDecl_ExportedEvv()
extern template void exportedFuncTmpl<ExplicitDecl_Exported>();
       template void exportedFuncTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of an exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI21ExplicitInst_ExportedEvv()
template void exportedFuncTmpl<ExplicitInst_Exported>();

// Export specialization of an exported function template.
// MSC-DAG: define dllexport void @"\01??$exportedFuncTmpl@UExplicitSpec_Def_Exported@@@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z16exportedFuncTmplI25ExplicitSpec_Def_ExportedEvv()
template<> __declspec(dllexport) void exportedFuncTmpl<ExplicitSpec_Def_Exported>() {}

// MSC-DAG: define weak_odr dllexport void @"\01??$exportedFuncTmpl@UExplicitSpec_InlineDef_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z16exportedFuncTmplI31ExplicitSpec_InlineDef_ExportedEvv()
template<> __declspec(dllexport) inline void exportedFuncTmpl<ExplicitSpec_InlineDef_Exported>() {}

// Not exporting specialization of an exported function template without
// explicit dllexport.
// MSC-DAG: define void @"\01??$exportedFuncTmpl@UExplicitSpec_NotExported@@@@YAXXZ"()
// GNU-DAG: define void @_Z16exportedFuncTmplI24ExplicitSpec_NotExportedEvv()
template<> void exportedFuncTmpl<ExplicitSpec_NotExported>() {}


// Export explicit instantiation declaration of a non-exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitDecl_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI21ExplicitDecl_ExportedEvv()
extern template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();
       template __declspec(dllexport) void funcTmpl<ExplicitDecl_Exported>();

// Export explicit instantiation definition of a non-exported function template.
// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitInst_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI21ExplicitInst_ExportedEvv()
template __declspec(dllexport) void funcTmpl<ExplicitInst_Exported>();

// Export specialization of a non-exported function template.
// MSC-DAG: define dllexport void @"\01??$funcTmpl@UExplicitSpec_Def_Exported@@@@YAXXZ"()
// GNU-DAG: define dllexport void @_Z8funcTmplI25ExplicitSpec_Def_ExportedEvv()
template<> __declspec(dllexport) void funcTmpl<ExplicitSpec_Def_Exported>() {}

// MSC-DAG: define weak_odr dllexport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Exported@@@@YAXXZ"()
// GNU-DAG: define weak_odr dllexport void @_Z8funcTmplI31ExplicitSpec_InlineDef_ExportedEvv()
template<> __declspec(dllexport) inline void funcTmpl<ExplicitSpec_InlineDef_Exported>() {}



//===----------------------------------------------------------------------===//
// Precedence
//===----------------------------------------------------------------------===//

// dllexport takes precedence over the dllimport if both are specified.
// MSC-DAG: @"\01?PrecedenceGlobal1A@@3HA" = dllexport global i32 0, align 4
// MSC-DAG: @"\01?PrecedenceGlobal1B@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal1A            = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal1B            = dllexport global i32 0, align 4
__attribute__((dllimport, dllexport))       int PrecedenceGlobal1A; // dllimport ignored
__declspec(dllimport) __declspec(dllexport) int PrecedenceGlobal1B; // dllimport ignored

// MSC-DAG: @"\01?PrecedenceGlobal2A@@3HA" = dllexport global i32 0, align 4
// MSC-DAG: @"\01?PrecedenceGlobal2B@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal2A            = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobal2B            = dllexport global i32 0, align 4
__attribute__((dllexport, dllimport))       int PrecedenceGlobal2A; // dllimport ignored
__declspec(dllexport) __declspec(dllimport) int PrecedenceGlobal2B; // dllimport ignored

// MSC-DAG: @"\01?PrecedenceGlobalRedecl1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalRedecl1            = dllexport global i32 0, align 4
__declspec(dllexport) extern int PrecedenceGlobalRedecl1;
__declspec(dllimport)        int PrecedenceGlobalRedecl1 = 0;

// MSC-DAG: @"\01?PrecedenceGlobalRedecl2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalRedecl2            = dllexport global i32 0, align 4
__declspec(dllimport) extern int PrecedenceGlobalRedecl2;
__declspec(dllexport)        int PrecedenceGlobalRedecl2;

// MSC-DAG: @"\01?PrecedenceGlobalMixed1@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalMixed1            = dllexport global i32 0, align 4
__attribute__((dllexport)) extern int PrecedenceGlobalMixed1;
__declspec(dllimport)             int PrecedenceGlobalMixed1 = 0;

// MSC-DAG: @"\01?PrecedenceGlobalMixed2@@3HA" = dllexport global i32 0, align 4
// GNU-DAG: @PrecedenceGlobalMixed2            = dllexport global i32 0, align 4
__attribute__((dllimport)) extern int PrecedenceGlobalMixed2;
__declspec(dllexport)             int PrecedenceGlobalMixed2;

// MSC-DAG: define dllexport void @"\01?precedence1A@@YAXXZ"
// MSC-DAG: define dllexport void @"\01?precedence1B@@YAXXZ"
// GNU-DAG: define dllexport void @_Z12precedence1Av()
// GNU-DAG: define dllexport void @_Z12precedence1Bv()
void __attribute__((dllimport, dllexport))       precedence1A() {}
void __declspec(dllimport) __declspec(dllexport) precedence1B() {}

// MSC-DAG: define dllexport void @"\01?precedence2A@@YAXXZ"
// MSC-DAG: define dllexport void @"\01?precedence2B@@YAXXZ"
// GNU-DAG: define dllexport void @_Z12precedence2Av()
// GNU-DAG: define dllexport void @_Z12precedence2Bv()
void __attribute__((dllexport, dllimport))       precedence2A() {}
void __declspec(dllexport) __declspec(dllimport) precedence2B() {}

// MSC-DAG: define dllexport void @"\01?precedenceRedecl1@@YAXXZ"
// GNU-DAG: define dllexport void @_Z17precedenceRedecl1v()
void __declspec(dllimport) precedenceRedecl1();
void __declspec(dllexport) precedenceRedecl1() {}

// MSC-DAG: define dllexport void @"\01?precedenceRedecl2@@YAXXZ"
// GNU-DAG: define dllexport void @_Z17precedenceRedecl2v()
void __declspec(dllexport) precedenceRedecl2();
void __declspec(dllimport) precedenceRedecl2() {}



//===----------------------------------------------------------------------===//
// Classes
//===----------------------------------------------------------------------===//

struct S {
  void __declspec(dllexport) a() {}
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?a@S@@QAEXXZ"

  struct T {
    void __declspec(dllexport) a() {}
    // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?a@T@S@@QAEXXZ"
  };
};

struct CtorWithClosure {
  __declspec(dllexport) CtorWithClosure(...) {}
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FCtorWithClosure@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// M32-DAG:   %[[this_addr:.*]] = alloca %struct.CtorWithClosure*, align 4
// M32-DAG:   store %struct.CtorWithClosure* %this, %struct.CtorWithClosure** %[[this_addr]], align 4
// M32-DAG:   %[[this:.*]] = load %struct.CtorWithClosure*, %struct.CtorWithClosure** %[[this_addr]]
// M32-DAG:   call %struct.CtorWithClosure* (%struct.CtorWithClosure*, ...) @"\01??0CtorWithClosure@@QAA@ZZ"(%struct.CtorWithClosure* %[[this]])
// M32-DAG:   ret void
};

#define DELETE_IMPLICIT_MEMBERS(ClassName) \
    ClassName(ClassName &&) = delete; \
    ClassName(ClassName &) = delete; \
    ~ClassName() = delete; \
    ClassName &operator=(ClassName &) = delete

struct __declspec(dllexport) ClassWithClosure {
  DELETE_IMPLICIT_MEMBERS(ClassWithClosure);
  ClassWithClosure(...) {}
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FClassWithClosure@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// M32-DAG:   %[[this_addr:.*]] = alloca %struct.ClassWithClosure*, align 4
// M32-DAG:   store %struct.ClassWithClosure* %this, %struct.ClassWithClosure** %[[this_addr]], align 4
// M32-DAG:   %[[this:.*]] = load %struct.ClassWithClosure*, %struct.ClassWithClosure** %[[this_addr]]
// M32-DAG:   call %struct.ClassWithClosure* (%struct.ClassWithClosure*, ...) @"\01??0ClassWithClosure@@QAA@ZZ"(%struct.ClassWithClosure* %[[this]])
// M32-DAG:   ret void
};

struct __declspec(dllexport) NestedOuter {
  DELETE_IMPLICIT_MEMBERS(NestedOuter);
  NestedOuter(void *p = 0) {}
  struct __declspec(dllexport) NestedInner {
    DELETE_IMPLICIT_MEMBERS(NestedInner);
    NestedInner(void *p = 0) {}
  };
};

// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FNestedOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FNestedInner@NestedOuter@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

template <typename T>
struct SomeTemplate {
  SomeTemplate(T o = T()) : o(o) {}
  T o;
};
// MSVC2015-DAG: define weak_odr dllexport {{.+}} @"\01??4?$SomeTemplate@H@@Q{{.+}}@$$Q{{.+}}@@Z"
// MSVC2013-DAG: define weak_odr dllexport {{.+}} @"\01??4?$SomeTemplate@H@@Q{{.+}}0@A{{.+}}0@@Z"
struct __declspec(dllexport) InheritFromTemplate : SomeTemplate<int> {};

// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_F?$SomeTemplate@H@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

namespace PR23801 {
template <typename>
struct S {
  ~S() {}
};
struct A {
  A(int);
  S<int> s;
};
struct __declspec(dllexport) B {
  B(A = 0) {}
};

}
//
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FB@PR23801@@QAEXXZ"({{.*}}) {{#[0-9]+}} comdat

struct __declspec(dllexport) T {
  // Copy assignment operator:
  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.T* @"\01??4T@@QAEAAU0@ABU0@@Z"

  // Explicitly defaulted copy constructur:
  T(const T&) = default;
  // M32MSVC2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.T* @"\01??0T@@QAE@ABU0@@Z"

  void a() {}
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?a@T@@QAEXXZ"

  static int b;
  // M32-DAG: @"\01?b@T@@2HA" = external dllexport global i32

  static int c;
  // M32-DAG: @"\01?c@T@@2HA" = dllexport global i32 0, align 4
};

USEVAR(T::b)
int T::c;

// Export template class with static member variable
// MSC-DAG: @"\01?StaticClassVarExpTmplClass@?$TmplClass@H@@2HA" = weak_odr dllexport global i32 0, comdat, align 4
// GNU-DAG: @_ZN9TmplClassIiE26StaticClassVarExpTmplClassE = weak_odr dllexport global i32 0, comdat, align 4
template<typename T>
struct __declspec(dllexport) TmplClass
{
  static T StaticClassVarExpTmplClass;
};

template<typename T>
T TmplClass<T>::StaticClassVarExpTmplClass;

// Export a definition of a template function.
// MSC-DAG: define weak_odr dllexport i32 @"\01??$TypeFunTmpl@H@@YAHH@Z"
// GNU-DAG: define weak_odr dllexport i32 @_Z11TypeFunTmplIiET_S0_
template<typename T>
T __declspec(dllexport) TypeFunTmpl(T t) { return t + t; }

// Instantiate the exported template class and the exported template function.
int useExportedTmplStaticAndFun()
{
  return TmplClass<int>::StaticClassVarExpTmplClass + TypeFunTmpl<int>(10);
}

template <typename T> struct __declspec(dllexport) U { void foo() {} };
struct __declspec(dllexport) V : public U<int> { };
// U<int>'s assignment operator is emitted.
// M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.U* @"\01??4?$U@H@@QAEAAU0@ABU0@@Z"

struct __declspec(dllexport) W { virtual void foo(); };
void W::foo() {}
// Default ctor:
// M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.W* @"\01??0W@@QAE@XZ"
// Copy ctor:
// M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.W* @"\01??0W@@QAE@ABU0@@Z"
// vftable:
// M32-DAG: [[W_VTABLE:@.*]] = private unnamed_addr constant [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4W@@6B@" to i8*), i8* bitcast (void (%struct.W*)* @"\01?foo@W@@UAEXXZ" to i8*)], comdat($"\01??_7W@@6B@")
// M32-DAG: @"\01??_7W@@6B@" = dllexport unnamed_addr alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* [[W_VTABLE]], i32 0, i32 1)
// G32-DAG: @_ZTV1W = dllexport unnamed_addr constant [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1W to i8*), i8* bitcast (void (%struct.W*)* @_ZN1W3fooEv to i8*)]

struct __declspec(dllexport) X : public virtual W {};
// vbtable:
// M32-DAG: @"\01??_8X@@7B@" = weak_odr dllexport unnamed_addr constant [2 x i32] [i32 0, i32 4]

struct __declspec(dllexport) Y {
  // Move assignment operator:
  // MSVC2015-DAG: define weak_odr dllexport {{.+}} @"\01??4Y@@Q{{.+}}@$$Q{{.+}}@@Z"
  // MSVC2013-DAG: define weak_odr dllexport {{.+}} @"\01??4Y@@Q{{.+}}0@A{{.+}}0@@Z"

  int x;
};

struct __declspec(dllexport) Z { virtual ~Z() {} };
// The scalar deleting dtor does not get exported:
// M32-DAG: define linkonce_odr x86_thiscallcc i8* @"\01??_GZ@@UAEPAXI@Z"


// The user-defined dtor does get exported:
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??1Z@@UAE@XZ"

namespace UseDtorAlias {
  struct __declspec(dllexport) A { ~A(); };
  struct __declspec(dllexport) B : A { ~B(); };
  A::~A() { }
  B::~B() { }
  // Emit a alias definition of B's constructor.
  // M32-DAG: @"\01??1B@UseDtorAlias@@QAE@XZ" = dllexport alias {{.*}} @"\01??1A@UseDtorAlias@@QAE@XZ"
}

struct __declspec(dllexport) DefaultedCtorsDtors {
  DefaultedCtorsDtors() = default;
  // M32MSVC2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.DefaultedCtorsDtors* @"\01??0DefaultedCtorsDtors@@QAE@XZ"
  ~DefaultedCtorsDtors() = default;
  // M32MSVC2013-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??1DefaultedCtorsDtors@@QAE@XZ"
};

// Export defaulted member function definitions declared inside class.
struct __declspec(dllexport) ExportDefaultedInclassDefs {
  ExportDefaultedInclassDefs() = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M64VS2013-DAG: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M64VS2015-NOT: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)

  ~ExportDefaultedInclassDefs() = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??1ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M64VS2013-DAG: define weak_odr dllexport                void @"\01??1ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc void @"\01??1ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M64VS2015-NOT: define weak_odr dllexport                void @"\01??1ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* %this)

  ExportDefaultedInclassDefs(const ExportDefaultedInclassDefs&) = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64VS2013-DAG: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64VS2015-NOT: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))

  ExportDefaultedInclassDefs& operator=(const ExportDefaultedInclassDefs&) = default;
  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedInclassDefs* @"\01??4ExportDefaultedInclassDefs@@QAEAAU0@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedInclassDefs* @"\01??4ExportDefaultedInclassDefs@@QEAAAEAU0@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
};

namespace ReferencedInlineMethodInNestedClass {
  struct __declspec(dllexport) S {
    void foo() {
      t->bar();
    }
    struct T {
      void bar() {}
    };
    T *t;
  };
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?foo@S@ReferencedInlineMethodInNestedClass@@QAEXXZ"
  // M32-DAG: define linkonce_odr x86_thiscallcc void @"\01?bar@T@S@ReferencedInlineMethodInNestedClass@@QAEXXZ"
}

// MS ignores DLL attributes on partial specializations.
template <typename T> struct PartiallySpecializedClassTemplate {};
template <typename T> struct __declspec(dllexport) PartiallySpecializedClassTemplate<T*> { void f(); };
template <typename T> void PartiallySpecializedClassTemplate<T*>::f() {}
USEMEMFUNC(PartiallySpecializedClassTemplate<void*>, f);
// M32-DAG: define linkonce_odr x86_thiscallcc void @"\01?f@?$PartiallySpecializedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN33PartiallySpecializedClassTemplateIPvE1fEv

// Attributes on explicit specializations are honored.
template <typename T> struct ExplicitlySpecializedClassTemplate {};
template <> struct __declspec(dllexport) ExplicitlySpecializedClassTemplate<void*> { void f(); };
void ExplicitlySpecializedClassTemplate<void*>::f() {}
USEMEMFUNC(ExplicitlySpecializedClassTemplate<void*>, f);
// M32-DAG: define dllexport x86_thiscallcc void @"\01?f@?$ExplicitlySpecializedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: define dllexport x86_thiscallcc void @_ZN34ExplicitlySpecializedClassTemplateIPvE1fEv

// MS inherits DLL attributes to partial specializations.
template <typename T> struct __declspec(dllexport) PartiallySpecializedExportedClassTemplate {};
template <typename T> struct PartiallySpecializedExportedClassTemplate<T*> { void f() {} };
USEMEMFUNC(PartiallySpecializedExportedClassTemplate<void*>, f);
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$PartiallySpecializedExportedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN41PartiallySpecializedExportedClassTemplateIPvE1fEv

// MS ignores DLL attributes on partial specializations; inheritance still works though.
template <typename T> struct __declspec(dllexport) PartiallySpecializedExportedClassTemplate2 {};
template <typename T> struct __declspec(dllimport) PartiallySpecializedExportedClassTemplate2<T*> { void f(); };
template <typename T> void PartiallySpecializedExportedClassTemplate2<T*>::f() {}
USEMEMFUNC(PartiallySpecializedExportedClassTemplate2<void*>, f);
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$PartiallySpecializedExportedClassTemplate2@PAX@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN42PartiallySpecializedExportedClassTemplate2IPvE1fEv

// Attributes on the instantiation take precedence over attributes on the template.
template <typename T> struct __declspec(dllimport) ExplicitlyInstantiatedWithDifferentAttr { void f() {} };
template struct __declspec(dllexport) ExplicitlyInstantiatedWithDifferentAttr<int>;
USEMEMFUNC(ExplicitlyInstantiatedWithDifferentAttr<int>, f);
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$ExplicitlyInstantiatedWithDifferentAttr@H@@QAEXXZ"

// Don't create weak dllexport aliases. (PR21373)
struct NonExportedBaseClass {
  virtual ~NonExportedBaseClass();
};
NonExportedBaseClass::~NonExportedBaseClass() {}

struct __declspec(dllexport) ExportedDerivedClass : NonExportedBaseClass {};
// M32-DAG: weak_odr dllexport x86_thiscallcc void @"\01??1ExportedDerivedClass@@UAE@XZ"

// Do not assert about generating code for constexpr functions twice during explicit instantiation (PR21718).
template <typename T> struct ExplicitInstConstexprMembers {
  // Copy assignment operator
  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable(1) %struct.ExplicitInstConstexprMembers* @"\01??4?$ExplicitInstConstexprMembers@X@@QAEAAU0@ABU0@@Z"

  constexpr ExplicitInstConstexprMembers() {}
  // M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExplicitInstConstexprMembers* @"\01??0?$ExplicitInstConstexprMembers@X@@QAE@XZ"

  ExplicitInstConstexprMembers(const ExplicitInstConstexprMembers&) = default;
  // M32MSVC2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExplicitInstConstexprMembers* @"\01??0?$ExplicitInstConstexprMembers@X@@QAE@ABU0@@Z"

  constexpr int f() const { return 42; }
  // M32-DAG: define weak_odr dllexport x86_thiscallcc i32 @"\01?f@?$ExplicitInstConstexprMembers@X@@QBEHXZ"
};
template struct __declspec(dllexport) ExplicitInstConstexprMembers<void>;

template <typename T> struct ExplicitInstantiationDeclTemplate { void f() {} };
extern template struct __declspec(dllexport) ExplicitInstantiationDeclTemplate<int>;
USEMEMFUNC(ExplicitInstantiationDeclTemplate<int>, f);
// M32-DAG: {{declare|define available_externally}} x86_thiscallcc void @"\01?f@?$ExplicitInstantiationDeclTemplate@H@@QAEXXZ"

template <typename T> struct __declspec(dllexport) ExplicitInstantiationDeclExportedTemplate { void f() {} };
extern template struct ExplicitInstantiationDeclExportedTemplate<int>;
USEMEMFUNC(ExplicitInstantiationDeclExportedTemplate<int>, f);
// M32-DAG: {{declare|define available_externally}} x86_thiscallcc void @"\01?f@?$ExplicitInstantiationDeclExportedTemplate@H@@QAEXXZ"

template <typename T> struct ExplicitInstantiationDeclExportedDefTemplate { void f() {} ExplicitInstantiationDeclExportedDefTemplate() {} };
extern template struct ExplicitInstantiationDeclExportedDefTemplate<int>;
template struct __declspec(dllexport) ExplicitInstantiationDeclExportedDefTemplate<int>;
USEMEMFUNC(ExplicitInstantiationDeclExportedDefTemplate<int>, f);
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$ExplicitInstantiationDeclExportedDefTemplate@H@@QAEXXZ"
// M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExplicitInstantiationDeclExportedDefTemplate* @"\01??0?$ExplicitInstantiationDeclExportedDefTemplate@H@@QAE@XZ"
// G32-DAG: define weak_odr x86_thiscallcc void @_ZN44ExplicitInstantiationDeclExportedDefTemplateIiE1fEv

namespace { struct InternalLinkageType {}; }
struct __declspec(dllexport) PR23308 {
  void f(InternalLinkageType*);
};
void PR23308::f(InternalLinkageType*) {}
long use(PR23308* p) { p->f(nullptr); }
// M32-DAG: define internal x86_thiscallcc void @"\01?f@PR23308@@QAEXPAUInternalLinkageType@?A@@@Z"

template <typename T> struct PR23770BaseTemplate { void f() {} };
template <typename T> struct PR23770DerivedTemplate : PR23770BaseTemplate<int> {};
extern template struct PR23770DerivedTemplate<int>;
template struct __declspec(dllexport) PR23770DerivedTemplate<int>;
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$PR23770BaseTemplate@H@@QAEXXZ"

namespace InClassInits {

struct __declspec(dllexport) S {
  int x = 42;
};
// M32-DAG: define weak_odr dllexport x86_thiscallcc %"struct.InClassInits::S"* @"\01??0S@InClassInits@@QAE@XZ"

// dllexport an already instantiated class template.
template <typename T> struct Base {
  int x = 42;
};
Base<int> base;
struct __declspec(dllexport) T : Base<int> { };
// M32-DAG: define weak_odr dllexport x86_thiscallcc %"struct.InClassInits::Base"* @"\01??0?$Base@H@InClassInits@@QAE@XZ"

struct A { A(int); };
struct __declspec(dllexport) U {
  // Class with both default constructor closure and in-class initializer.
  U(A = 0) {}
  int x = 0;
};
// M32-DAG: define weak_odr dllexport x86_thiscallcc %"struct.InClassInits::U"* @"\01??0U@InClassInits@@QAE@UA@1@@Z"

struct Evil {
  template <typename T> struct Base {
    int x = 0;
  };
  struct S : Base<int> {};
  // The already instantiated Base<int> becomes dllexported below, but the
  // in-class initializer for Base<>::x still hasn't been parsed, so emitting
  // the default ctor must still be delayed.
  struct __declspec(dllexport) T : Base<int> {};
};
// M32-DAG: define weak_odr dllexport x86_thiscallcc %"struct.InClassInits::Evil::Base"* @"\01??0?$Base@H@Evil@InClassInits@@QAE@XZ"

template <typename T> struct Foo {};
template <typename T> struct Bar {
  Bar<T> &operator=(Foo<T>) {}
};
struct __declspec(dllexport) Baz {
  Bar<int> n;
};
// After parsing Baz, in ActOnFinishCXXNonNestedClass we would synthesize
// Baz's operator=, causing instantiation of Foo<int> after which
// ActOnFinishCXXNonNestedClass is called, and we would bite our own tail.
// M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable(1) %"struct.InClassInits::Baz"* @"\01??4Baz@InClassInits@@QAEAAU01@ABU01@@Z"
}

// We had an issue where instantiating A would force emission of B's delayed
// exported methods.
namespace pr26490 {
template <typename T> struct A { };
struct __declspec(dllexport) B {
  B(int = 0) {}
  A<int> m_fn1() {}
};
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??_FB@pr26490@@QAEXXZ"
}

// dllexport trumps dllexport on an explicit instantiation.
template <typename T> struct ExplicitInstantiationTwoAttributes { void f() {} };
template struct __declspec(dllexport) __declspec(dllimport) ExplicitInstantiationTwoAttributes<int>;
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?f@?$ExplicitInstantiationTwoAttributes@H@@QAEXXZ"


//===----------------------------------------------------------------------===//
// Classes with template base classes
//===----------------------------------------------------------------------===//

template <typename T> struct ClassTemplate { void func(); };
template <typename T> void ClassTemplate<T>::func() {}
template <typename T> struct __declspec(dllexport) ExportedClassTemplate { void func(); };
template <typename T> void ExportedClassTemplate<T>::func() {}
template <typename T> struct __declspec(dllimport) ImportedClassTemplate { void func(); };
template <typename T> void ImportedClassTemplate<T>::func() {}

template <typename T> struct ExplicitlySpecializedTemplate { void func() {} };
template <> struct ExplicitlySpecializedTemplate<int> { void func(); };
void ExplicitlySpecializedTemplate<int>::func() {}
template <typename T> struct ExplicitlyExportSpecializedTemplate { void func() {} };
template <> struct __declspec(dllexport) ExplicitlyExportSpecializedTemplate<int> { void func(); };
void ExplicitlyExportSpecializedTemplate<int>::func() {}
template <typename T> struct ExplicitlyImportSpecializedTemplate { void func(); };
template <> struct __declspec(dllimport) ExplicitlyImportSpecializedTemplate<int> { void func(); };

template <typename T> struct ExplicitlyInstantiatedTemplate { void func(); };
template <typename T> void ExplicitlyInstantiatedTemplate<T>::func() {}
template struct ExplicitlyInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyExportInstantiatedTemplate { void func(); };
template <typename T> void ExplicitlyExportInstantiatedTemplate<T>::func() {}
template struct __declspec(dllexport) ExplicitlyExportInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyImportInstantiatedTemplate { void func(); };
template struct __declspec(dllimport) ExplicitlyImportInstantiatedTemplate<int>;


// MS: ClassTemplate<int> gets exported.
struct __declspec(dllexport) DerivedFromTemplate : public ClassTemplate<int> {};
USEMEMFUNC(DerivedFromTemplate, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ClassTemplate@H@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIiE4funcEv

// ExportedTemplate is explicitly exported.
struct __declspec(dllexport) DerivedFromExportedTemplate : public ExportedClassTemplate<int> {};
USEMEMFUNC(DerivedFromExportedTemplate, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExportedClassTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN21ExportedClassTemplateIiE4funcEv

// ImportedClassTemplate is explicitly imported.
struct __declspec(dllexport) DerivedFromImportedTemplate : public ImportedClassTemplate<int> {};
USEMEMFUNC(DerivedFromImportedTemplate, func)
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?func@?$ImportedClassTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN21ImportedClassTemplateIiE4funcEv

// Base class already implicitly instantiated without dll attribute.
struct DerivedFromTemplateD : public ClassTemplate<double> {};
struct __declspec(dllexport) DerivedFromTemplateD2 : public ClassTemplate<double> {};
USEMEMFUNC(DerivedFromTemplateD2, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ClassTemplate@N@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIdE4funcEv

// MS: Base class already instantiated with different dll attribute.
struct __declspec(dllimport) DerivedFromTemplateB : public ClassTemplate<bool> {};
struct __declspec(dllexport) DerivedFromTemplateB2 : public ClassTemplate<bool> {};
USEMEMFUNC(DerivedFromTemplateB2, func)
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?func@?$ClassTemplate@_N@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIbE4funcEv

// Base class already specialized without dll attribute.
struct __declspec(dllexport) DerivedFromExplicitlySpecializedTemplate : public ExplicitlySpecializedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlySpecializedTemplate, func)
// M32-DAG: define x86_thiscallcc void @"\01?func@?$ExplicitlySpecializedTemplate@H@@QAEXXZ"
// G32-DAG: define x86_thiscallcc void @_ZN29ExplicitlySpecializedTemplateIiE4funcEv

// Base class alredy specialized with export attribute.
struct __declspec(dllexport) DerivedFromExplicitlyExportSpecializedTemplate : public ExplicitlyExportSpecializedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlyExportSpecializedTemplate, func)
// M32-DAG: define dllexport x86_thiscallcc void @"\01?func@?$ExplicitlyExportSpecializedTemplate@H@@QAEXXZ"
// G32-DAG: define dllexport x86_thiscallcc void @_ZN35ExplicitlyExportSpecializedTemplateIiE4funcEv

// Base class already specialized with import attribute.
struct __declspec(dllexport) DerivedFromExplicitlyImportSpecializedTemplate : public ExplicitlyImportSpecializedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlyImportSpecializedTemplate, func)
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?func@?$ExplicitlyImportSpecializedTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN35ExplicitlyImportSpecializedTemplateIiE4funcEv

// Base class already instantiated without dll attribute.
struct __declspec(dllexport) DerivedFromExplicitlyInstantiatedTemplate : public ExplicitlyInstantiatedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlyInstantiatedTemplate, func)
// M32-DAG: define weak_odr x86_thiscallcc void @"\01?func@?$ExplicitlyInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr x86_thiscallcc void @_ZN30ExplicitlyInstantiatedTemplateIiE4funcEv

// Base class already instantiated with export attribute.
struct __declspec(dllexport) DerivedFromExplicitlyExportInstantiatedTemplate : public ExplicitlyExportInstantiatedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlyExportInstantiatedTemplate, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExplicitlyExportInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN36ExplicitlyExportInstantiatedTemplateIiE4funcEv

// Base class already instantiated with import attribute.
struct __declspec(dllexport) DerivedFromExplicitlyImportInstantiatedTemplate : public ExplicitlyImportInstantiatedTemplate<int> {};
USEMEMFUNC(DerivedFromExplicitlyImportInstantiatedTemplate, func)
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?func@?$ExplicitlyImportInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN36ExplicitlyImportInstantiatedTemplateIiE4funcEv

// MS: A dll attribute propagates through multiple levels of instantiation.
template <typename T> struct TopClass { void func() {} };
template <typename T> struct MiddleClass : public TopClass<T> { };
struct __declspec(dllexport) BottomClass : public MiddleClass<int> { };
USEMEMFUNC(BottomClass, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$TopClass@H@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN8TopClassIiE4funcEv

template <typename T> struct ExplicitInstantiationDeclTemplateBase { void func() {} };
extern template struct ExplicitInstantiationDeclTemplateBase<int>;
struct __declspec(dllexport) DerivedFromExplicitInstantiationDeclTemplateBase : public ExplicitInstantiationDeclTemplateBase<int> {};
template struct ExplicitInstantiationDeclTemplateBase<int>;
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExplicitInstantiationDeclTemplateBase@H@@QAEXXZ"
// G32-DAG: define weak_odr x86_thiscallcc void @_ZN37ExplicitInstantiationDeclTemplateBaseIiE4funcEv

// PR26076
struct LayerSelectionBound;
template <typename> struct Selection {};
typedef Selection<LayerSelectionBound> LayerSelection;
struct LayerImpl;
struct __declspec(dllexport) LayerTreeImpl {
  struct __declspec(dllexport) ElementLayers {
    LayerImpl *main = nullptr;
  };
  LayerSelection foo;
};
// M32-DAG: define weak_odr dllexport x86_thiscallcc %"struct.LayerTreeImpl::ElementLayers"* @"\01??0ElementLayers@LayerTreeImpl@@QAE@XZ"
// M64-DAG: define weak_odr dllexport %"struct.LayerTreeImpl::ElementLayers"* @"\01??0ElementLayers@LayerTreeImpl@@QEAA@XZ"

class __declspec(dllexport) ACE_Shared_Object {
public:
  virtual ~ACE_Shared_Object();
};
class __declspec(dllexport) ACE_Service_Object : public ACE_Shared_Object {};
// Implicit move constructor declaration.
// MSVC2015-DAG: define weak_odr dllexport {{.+}}ACE_Service_Object@@Q{{.+}}@$$Q
// The declarations should not be exported.
// MSVC2013-NOT: define weak_odr dllexport {{.+}}ACE_Service_Object@@Q{{.+}}@$$Q
