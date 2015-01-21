// RUN: %clang_cc1 -triple i686-windows-msvc   -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s -DMSABI -w | FileCheck --check-prefix=MSC --check-prefix=M32 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s -DMSABI -w | FileCheck --check-prefix=MSC --check-prefix=M64 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s         -w | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s         -w | FileCheck --check-prefix=GNU --check-prefix=G64 %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -fno-rtti -emit-llvm -std=c++1y -O1 -o - %s -DMSABI -w | FileCheck --check-prefix=MO1 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fno-rtti -emit-llvm -std=c++1y -O1 -o - %s         -w | FileCheck --check-prefix=GO1 %s

// CHECK-NOT doesn't play nice with CHECK-DAG, so use separate run lines.
// RUN: %clang_cc1 -triple i686-windows-msvc   -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s -DMSABI -w | FileCheck --check-prefix=MSC2 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s         -w | FileCheck --check-prefix=GNU2 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Imported {};
struct ImplicitInst_NotImported {};
struct ExplicitDecl_Imported {};
struct ExplicitInst_Imported {};
struct ExplicitSpec_Imported {};
struct ExplicitSpec_Def_Imported {};
struct ExplicitSpec_InlineDef_Imported {};
struct ExplicitSpec_NotImported {};

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEVARTYPE(type, var) type UNIQ(use)() { return var; }
#define USEVAR(var) USEVARTYPE(int, var)
#define USE(func) void UNIQ(use)() { func(); }
#define USEMEMFUNC(class, func) void (class::*UNIQ(use)())() { return &class::func; }
#define USECLASS(class) void UNIQ(USE)() { class x; }
#define USECOPYASSIGN(class) class& (class::*UNIQ(use)())(class&) { return &class::operator=; }
#define USEMOVEASSIGN(class) class& (class::*UNIQ(use)())(class&&) { return &class::operator=; }

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
// MSC-DAG: @"\01?ExternGlobalDecl@@3HA" = external dllimport global i32
// GNU-DAG: @ExternGlobalDecl            = external dllimport global i32
__declspec(dllimport) extern int ExternGlobalDecl;
USEVAR(ExternGlobalDecl)

// dllimport implies a declaration.
// MSC-DAG: @"\01?GlobalDecl@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalDecl            = external dllimport global i32
__declspec(dllimport) int GlobalDecl;
USEVAR(GlobalDecl)

// Redeclarations
// MSC-DAG: @"\01?GlobalRedecl1@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl1            = external dllimport global i32
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;
USEVAR(GlobalRedecl1)

// MSC-DAG: @"\01?GlobalRedecl2a@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl2a            = external dllimport global i32
__declspec(dllimport) int GlobalRedecl2a;
__declspec(dllimport) int GlobalRedecl2a;
USEVAR(GlobalRedecl2a)

// M32-DAG: @"\01?GlobalRedecl2b@@3PAHA"   = external dllimport global i32*
// M64-DAG: @"\01?GlobalRedecl2b@@3PEAHEA" = external dllimport global i32*
// GNU-DAG: @GlobalRedecl2b                = external dllimport global i32*
int *__attribute__((dllimport)) GlobalRedecl2b;
int *__attribute__((dllimport)) GlobalRedecl2b;
USEVARTYPE(int*, GlobalRedecl2b)

// MSC-DAG: @"\01?GlobalRedecl2c@@3HA" = external dllimport global i32
// GNU-DAG: @GlobalRedecl2c            = external dllimport global i32
int GlobalRedecl2c __attribute__((dllimport));
int GlobalRedecl2c __attribute__((dllimport));
USEVAR(GlobalRedecl2c)

// NB: MSC issues a warning and makes GlobalRedecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// MSC-DAG: @"\01?GlobalRedecl3@@3HA" = external global i32
// GNU-DAG: @GlobalRedecl3            = external global i32
__declspec(dllimport) extern int GlobalRedecl3;
                      extern int GlobalRedecl3; // dllimport ignored
USEVAR(GlobalRedecl3)

// MSC-DAG: @"\01?ExternalGlobal@ns@@3HA" = external dllimport global i32
// GNU-DAG: @_ZN2ns14ExternalGlobalE      = external dllimport global i32
namespace ns { __declspec(dllimport) int ExternalGlobal; }
USEVAR(ns::ExternalGlobal)

int f();
// MO1-DAG: @"\01?x@?1??inlineStaticLocalsFunc@@YAHXZ@4HA" = available_externally dllimport global i32 0
// MO1-DAG: @"\01??_B?1??inlineStaticLocalsFunc@@YAHXZ@51" = available_externally dllimport global i32 0
inline int __declspec(dllimport) inlineStaticLocalsFunc() {
  static int x = f();
  return x++;
};
USE(inlineStaticLocalsFunc);

// The address of a dllimport global cannot be used in constant initialization.
// M32-DAG: @"\01?arr@?1??initializationFunc@@YAPAHXZ@4QBQAHB" = internal global [1 x i32*] zeroinitializer
// GNU-DAG: @_ZZ18initializationFuncvE3arr = internal global [1 x i32*] zeroinitializer
int *initializationFunc() {
  static int *const arr[] = {&ExternGlobalDecl};
  return arr[0];
}
USE(initializationFunc);


//===----------------------------------------------------------------------===//
// Variable templates
//===----------------------------------------------------------------------===//

// Import declaration.
// MSC-DAG: @"\01??$ExternVarTmplDecl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z17ExternVarTmplDeclI21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) extern int ExternVarTmplDecl;
USEVAR(ExternVarTmplDecl<ImplicitInst_Imported>)

// dllimport implies a declaration.
// MSC-DAG: @"\01??$VarTmplDecl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z11VarTmplDeclI21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) int VarTmplDecl;
USEVAR(VarTmplDecl<ImplicitInst_Imported>)

// Redeclarations
// MSC-DAG: @"\01??$VarTmplRedecl1@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z14VarTmplRedecl1I21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;
USEVAR(VarTmplRedecl1<ImplicitInst_Imported>)

// MSC-DAG: @"\01??$VarTmplRedecl2@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z14VarTmplRedecl2I21ImplicitInst_ImportedE          = external dllimport global i32
template<typename T> __declspec(dllimport) int VarTmplRedecl2;
template<typename T> __declspec(dllimport) int VarTmplRedecl2;
USEVAR(VarTmplRedecl2<ImplicitInst_Imported>)

// MSC-DAG: @"\01??$VarTmplRedecl3@UImplicitInst_Imported@@@@3HA" = external global i32
// GNU-DAG: @_Z14VarTmplRedecl3I21ImplicitInst_ImportedE          = external global i32
template<typename T> __declspec(dllimport) extern int VarTmplRedecl3;
template<typename T>                       extern int VarTmplRedecl3; // dllimport ignored
USEVAR(VarTmplRedecl3<ImplicitInst_Imported>)


// MSC-DAG: @"\01??$ExternalVarTmpl@UImplicitInst_Imported@@@ns@@3HA" = external dllimport global i32
// GNU-DAG: @_ZN2ns15ExternalVarTmplI21ImplicitInst_ImportedEE        = external dllimport global i32
namespace ns { template<typename T> __declspec(dllimport) int ExternalVarTmpl; }
USEVAR(ns::ExternalVarTmpl<ImplicitInst_Imported>)


template<typename T> int VarTmpl;
template<typename T> __declspec(dllimport) int ImportedVarTmpl;

// Import implicit instantiation of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UImplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ImplicitInst_ImportedE          = external dllimport global i32
USEVAR(ImportedVarTmpl<ImplicitInst_Imported>)

// Import explicit instantiation declaration of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitDecl_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ExplicitDecl_ImportedE          = external dllimport global i32
extern template int ImportedVarTmpl<ExplicitDecl_Imported>;
USEVAR(ImportedVarTmpl<ExplicitDecl_Imported>)

// An explicit instantiation definition of an imported variable template cannot
// be imported because the template must be defined which is illegal.

// Import specialization of an imported variable template.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitSpec_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z15ImportedVarTmplI21ExplicitSpec_ImportedE          = external dllimport global i32
template<> __declspec(dllimport) int ImportedVarTmpl<ExplicitSpec_Imported>;
USEVAR(ImportedVarTmpl<ExplicitSpec_Imported>)

// Not importing specialization of an imported variable template without
// explicit dllimport.
// MSC-DAG: @"\01??$ImportedVarTmpl@UExplicitSpec_NotImported@@@@3HA" = global i32 0, align 4
// GNU-DAG: @_Z15ImportedVarTmplI24ExplicitSpec_NotImportedE          = global i32 0, align 4
template<> int ImportedVarTmpl<ExplicitSpec_NotImported>;
USEVAR(ImportedVarTmpl<ExplicitSpec_NotImported>)

// Import explicit instantiation declaration of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitDecl_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitDecl_ImportedE           = external dllimport global i32
extern template __declspec(dllimport) int VarTmpl<ExplicitDecl_Imported>;
USEVAR(VarTmpl<ExplicitDecl_Imported>)

// Import explicit instantiation definition of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitInst_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitInst_ImportedE           = external dllimport global i32
template __declspec(dllimport) int VarTmpl<ExplicitInst_Imported>;
USEVAR(VarTmpl<ExplicitInst_Imported>)

// Import specialization of a non-imported variable template.
// MSC-DAG: @"\01??$VarTmpl@UExplicitSpec_Imported@@@@3HA" = external dllimport global i32
// GNU-DAG: @_Z7VarTmplI21ExplicitSpec_ImportedE           = external dllimport global i32
template<> __declspec(dllimport) int VarTmpl<ExplicitSpec_Imported>;
USEVAR(VarTmpl<ExplicitSpec_Imported>)



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration.
// MSC-DAG: declare dllimport void @"\01?decl@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z4declv()
__declspec(dllimport) void decl();
USE(decl)

// extern "C"
// MSC-DAG: declare dllimport void @externC()
// GNU-DAG: declare dllimport void @externC()
extern "C" __declspec(dllimport) void externC();
USE(externC)

// Import inline function.
// MSC-DAG: declare dllimport void @"\01?inlineFunc@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z10inlineFuncv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineFunc@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z10inlineFuncv()
__declspec(dllimport) inline void inlineFunc() {}
USE(inlineFunc)

// MSC-DAG: declare dllimport void @"\01?inlineDecl@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z10inlineDeclv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineDecl@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z10inlineDeclv()
__declspec(dllimport) inline void inlineDecl();
                             void inlineDecl() {}
USE(inlineDecl)

// MSC-DAG: declare dllimport void @"\01?inlineDef@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z9inlineDefv()
// MO1-DAG: define available_externally dllimport void @"\01?inlineDef@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z9inlineDefv()
__declspec(dllimport) void inlineDef();
               inline void inlineDef() {}
USE(inlineDef)

// inline attributes
// MSC-DAG: declare dllimport void @"\01?noinline@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z8noinlinev()
__declspec(dllimport) __attribute__((noinline)) inline void noinline() {}
USE(noinline)

// MSC2-NOT: @"\01?alwaysInline@@YAXXZ"()
// GNU2-NOT: @_Z12alwaysInlinev()
__declspec(dllimport) __attribute__((always_inline)) inline void alwaysInline() {}
USE(alwaysInline)

// Redeclarations
// MSC-DAG: declare dllimport void @"\01?redecl1@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z7redecl1v()
__declspec(dllimport) void redecl1();
__declspec(dllimport) void redecl1();
USE(redecl1)

// NB: MSC issues a warning and makes redecl2/redecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// MSC-DAG: declare void @"\01?redecl2@@YAXXZ"()
// GNU-DAG: declare void @_Z7redecl2v()
__declspec(dllimport) void redecl2();
                      void redecl2();
USE(redecl2)

// MSC-DAG: define void @"\01?redecl3@@YAXXZ"()
// GNU-DAG: define void @_Z7redecl3v()
__declspec(dllimport) void redecl3();
                      void redecl3() {} // dllimport ignored
USE(redecl3)


// Friend functions
// MSC-DAG: declare dllimport void @"\01?friend1@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z7friend1v()
// MSC-DAG: declare           void @"\01?friend2@@YAXXZ"()
// GNU-DAG: declare           void @_Z7friend2v()
// MSC-DAG: define            void @"\01?friend3@@YAXXZ"()
// GNU-DAG: define            void @_Z7friend3v()
// MSC-DAG: declare           void @"\01?friend4@@YAXXZ"()
// GNU-DAG: declare           void @_Z7friend4v()
// MSC-DAG: declare dllimport void @"\01?friend5@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z7friend5v()

struct FuncFriend {
  friend __declspec(dllimport) void friend1();
  friend __declspec(dllimport) void friend2();
  friend __declspec(dllimport) void friend3();
};
__declspec(dllimport) void friend1();
                      void friend2(); // dllimport ignored
                      void friend3() {} // dllimport ignored

__declspec(dllimport) void friend4();
__declspec(dllimport) void friend5();
struct FuncFriendRedecl {
  friend void friend4(); // dllimport ignored
  friend void ::friend5();
};
USE(friend1)
USE(friend2)
USE(friend3)
USE(friend4)
USE(friend5)

// Implicit declarations can be redeclared with dllimport.
// MSC-DAG: declare dllimport noalias i8* @"\01??2@{{YAPAXI|YAPEAX_K}}@Z"(
// GNU-DAG: declare dllimport noalias i8* @_Znw{{[yj]}}(
__declspec(dllimport) void* operator new(__SIZE_TYPE__ n);
void UNIQ(use)() { ::operator new(42); }

// MSC-DAG: declare dllimport void @"\01?externalFunc@ns@@YAXXZ"()
// GNU-DAG: declare dllimport void @_ZN2ns12externalFuncEv()
namespace ns { __declspec(dllimport) void externalFunc(); }
USE(ns::externalFunc)



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Import function template declaration.
// MSC-DAG: declare dllimport void @"\01??$funcTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z12funcTmplDeclI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplDecl();
USE(funcTmplDecl<ImplicitInst_Imported>)

// Function template definitions cannot be imported.

// Import inline function template.
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15inlineFuncTmpl1I21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl1@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z15inlineFuncTmpl1I21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) inline void inlineFuncTmpl1() {}
USE(inlineFuncTmpl1<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl2@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15inlineFuncTmpl2I21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl2@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z15inlineFuncTmpl2I21ImplicitInst_ImportedEvv()
template<typename T> inline void __attribute__((dllimport)) inlineFuncTmpl2() {}
USE(inlineFuncTmpl2<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z18inlineFuncTmplDeclI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z18inlineFuncTmplDeclI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) inline void inlineFuncTmplDecl();
template<typename T>                              void inlineFuncTmplDecl() {}
USE(inlineFuncTmplDecl<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmplDef@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z17inlineFuncTmplDefI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmplDef@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z17inlineFuncTmplDefI21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void inlineFuncTmplDef();
template<typename T>                inline void inlineFuncTmplDef() {}
USE(inlineFuncTmplDef<ImplicitInst_Imported>)


// Redeclarations
// MSC-DAG: declare dllimport void @"\01??$funcTmplRedecl1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z15funcTmplRedecl1I21ImplicitInst_ImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl1();
template<typename T> __declspec(dllimport) void funcTmplRedecl1();
USE(funcTmplRedecl1<ImplicitInst_Imported>)

// MSC-DAG: declare void @"\01??$funcTmplRedecl2@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: declare void @_Z15funcTmplRedecl2I24ImplicitInst_NotImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl2();
template<typename T>                       void funcTmplRedecl2(); // dllimport ignored
USE(funcTmplRedecl2<ImplicitInst_NotImported>)

// MSC-DAG: define linkonce_odr void @"\01??$funcTmplRedecl3@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15funcTmplRedecl3I24ImplicitInst_NotImportedEvv()
template<typename T> __declspec(dllimport) void funcTmplRedecl3();
template<typename T>                       void funcTmplRedecl3() {} // dllimport ignored
USE(funcTmplRedecl3<ImplicitInst_NotImported>)


// Function template friends
// MSC-DAG: declare dllimport   void @"\01??$funcTmplFriend1@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport   void @_Z15funcTmplFriend1I21ImplicitInst_ImportedEvv()
// MSC-DAG: declare             void @"\01??$funcTmplFriend2@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: declare             void @_Z15funcTmplFriend2I24ImplicitInst_NotImportedEvv()
// MSC-DAG: define linkonce_odr void @"\01??$funcTmplFriend3@UImplicitInst_NotImported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15funcTmplFriend3I24ImplicitInst_NotImportedEvv()
// MSC-DAG: declare dllimport   void @"\01??$funcTmplFriend4@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z15funcTmplFriend4I21ImplicitInst_ImportedEvv()
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllimport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllimport) void funcTmplFriend2();
  template<typename T> friend __declspec(dllimport) void funcTmplFriend3();
  template<typename T> friend __declspec(dllimport) inline void funcTmplFriend4();
};
template<typename T> __declspec(dllimport) void funcTmplFriend1();
template<typename T>                       void funcTmplFriend2(); // dllimport ignored
template<typename T>                       void funcTmplFriend3() {} // dllimport ignored
template<typename T>                       inline void funcTmplFriend4() {}
USE(funcTmplFriend1<ImplicitInst_Imported>)
USE(funcTmplFriend2<ImplicitInst_NotImported>)
USE(funcTmplFriend3<ImplicitInst_NotImported>)
USE(funcTmplFriend4<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$externalFuncTmpl@UImplicitInst_Imported@@@ns@@YAXXZ"()
// GNU-DAG: declare dllimport void @_ZN2ns16externalFuncTmplI21ImplicitInst_ImportedEEvv()
namespace ns { template<typename T> __declspec(dllimport) void externalFuncTmpl(); }
USE(ns::externalFuncTmpl<ImplicitInst_Imported>)


template<typename T> void funcTmpl() {}
template<typename T> inline void inlineFuncTmpl() {}
template<typename T> __declspec(dllimport) void importedFuncTmplDecl();
template<typename T> __declspec(dllimport) inline void importedFuncTmpl() {}

// Import implicit instantiation of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z20importedFuncTmplDeclI21ImplicitInst_ImportedEvv()
USE(importedFuncTmplDecl<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UImplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z16importedFuncTmplI21ImplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UImplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z16importedFuncTmplI21ImplicitInst_ImportedEvv()
USE(importedFuncTmpl<ImplicitInst_Imported>)

// Import explicit instantiation declaration of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GNU-DAG: declare void @_Z16importedFuncTmplI21ExplicitDecl_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally void @_Z16importedFuncTmplI21ExplicitDecl_ImportedEvv()
extern template void importedFuncTmpl<ExplicitDecl_Imported>();
USE(importedFuncTmpl<ExplicitDecl_Imported>)

// Import explicit instantiation definition of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: define weak_odr void @_Z16importedFuncTmplI21ExplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define weak_odr void @_Z16importedFuncTmplI21ExplicitInst_ImportedEvv()
template void importedFuncTmpl<ExplicitInst_Imported>();
USE(importedFuncTmpl<ExplicitInst_Imported>)


// Import specialization of an imported function template.
// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z20importedFuncTmplDeclI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void importedFuncTmplDecl<ExplicitSpec_Imported>();
USE(importedFuncTmplDecl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void importedFuncTmplDecl<ExplicitSpec_Def_Imported>() {}
//USE(importedFuncTmplDecl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z20importedFuncTmplDeclI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmplDecl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z20importedFuncTmplDeclI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void importedFuncTmplDecl<ExplicitSpec_InlineDef_Imported>() {}
USE(importedFuncTmplDecl<ExplicitSpec_InlineDef_Imported>)


// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z16importedFuncTmplI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Imported>();
USE(importedFuncTmpl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Def_Imported>() {}
//USE(importedFuncTmpl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z16importedFuncTmplI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$importedFuncTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z16importedFuncTmplI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void importedFuncTmpl<ExplicitSpec_InlineDef_Imported>() {}
USE(importedFuncTmpl<ExplicitSpec_InlineDef_Imported>)


// Not importing specialization of an imported function template without
// explicit dllimport.
// MSC-DAG: define void @"\01??$importedFuncTmpl@UExplicitSpec_NotImported@@@@YAXXZ"()
// GNU-DAG: define void @_Z16importedFuncTmplI24ExplicitSpec_NotImportedEvv()
template<> void importedFuncTmpl<ExplicitSpec_NotImported>() {}
USE(importedFuncTmpl<ExplicitSpec_NotImported>)


// Import explicit instantiation declaration of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitDecl_ImportedEvv()
// GNU-DAG: declare void @_Z14inlineFuncTmplI21ExplicitDecl_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl@UExplicitDecl_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally void @_Z14inlineFuncTmplI21ExplicitDecl_ImportedEvv()
extern template __declspec(dllimport) void funcTmpl<ExplicitDecl_Imported>();
extern template __declspec(dllimport) void inlineFuncTmpl<ExplicitDecl_Imported>();
USE(funcTmpl<ExplicitDecl_Imported>)
USE(inlineFuncTmpl<ExplicitDecl_Imported>)


// Import explicit instantiation definition of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// MSC-DAG: declare dllimport void @"\01??$inlineFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitInst_ImportedEvv()
// GNU-DAG: define weak_odr void @_Z14inlineFuncTmplI21ExplicitInst_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$funcTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// MO1-DAG: define available_externally dllimport void @"\01??$inlineFuncTmpl@UExplicitInst_Imported@@@@YAXXZ"()
// GO1-DAG: define available_externally dllimport void @_Z8funcTmplI21ExplicitInst_ImportedEvv()
// GO1-DAG: define weak_odr void @_Z14inlineFuncTmplI21ExplicitInst_ImportedEvv()
template __declspec(dllimport) void funcTmpl<ExplicitInst_Imported>();
template __declspec(dllimport) void inlineFuncTmpl<ExplicitInst_Imported>();
USE(funcTmpl<ExplicitInst_Imported>)
USE(inlineFuncTmpl<ExplicitInst_Imported>)


// Import specialization of a non-imported function template.
// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_Imported@@@@YAXXZ"()
// GNU-DAG: declare dllimport void @_Z8funcTmplI21ExplicitSpec_ImportedEvv()
template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Imported>();
USE(funcTmpl<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
// MO1-DAG-FIXME: define available_externally dllimport void @"\01??$funcTmpl@UExplicitSpec_Def_Imported@@@@YAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Def_Imported>() {}
//USE(funcTmpl<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GNU-DAG: define linkonce_odr void @_Z8funcTmplI31ExplicitSpec_InlineDef_ImportedEvv()
// MO1-DAG: define available_externally dllimport void @"\01??$funcTmpl@UExplicitSpec_InlineDef_Imported@@@@YAXXZ"()
// GO1-DAG: define linkonce_odr void @_Z8funcTmplI31ExplicitSpec_InlineDef_ImportedEvv()
template<> __declspec(dllimport) inline void funcTmpl<ExplicitSpec_InlineDef_Imported>() {}
USE(funcTmpl<ExplicitSpec_InlineDef_Imported>)



//===----------------------------------------------------------------------===//
// Classes
//===----------------------------------------------------------------------===//

struct __declspec(dllimport) T {
  void a() {}
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"\01?a@T@@QAEXXZ"

  static int b;
  // MO1-DAG: @"\01?b@T@@2HA" = external dllimport global i32

  T& operator=(T&) = default;
  // MO1-DAG: define available_externally dllimport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.T* @"\01??4T@@QAEAAU0@AAU0@@Z"

  T& operator=(T&&) = default;
  // Note: Don't mark inline move operators dllimport because current MSVC versions don't export them.
  // MO1-DAG: define linkonce_odr x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.T* @"\01??4T@@QAEAAU0@$$QAU0@@Z"
};
USEMEMFUNC(T, a)
USEVAR(T::b)
USECOPYASSIGN(T)
USEMOVEASSIGN(T)

template <typename T> struct __declspec(dllimport) U { void foo() {} };
// MO1-DAG: define available_externally dllimport x86_thiscallcc void @"\01?foo@?$U@H@@QAEXXZ"
struct __declspec(dllimport) V : public U<int> { };
USEMEMFUNC(V, foo)

struct __declspec(dllimport) W { virtual void foo() {} };
USECLASS(W)
// vftable:
// MO1-DAG: @"\01??_7W@@6B@" = available_externally dllimport unnamed_addr constant [1 x i8*] [i8* bitcast (void (%struct.W*)* @"\01?foo@W@@UAEXXZ" to i8*)]
// GO1-DAG: @_ZTV1W = available_externally dllimport unnamed_addr constant [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.W*)* @_ZN1W3fooEv to i8*)]

struct __declspec(dllimport) KeyFuncClass {
  constexpr KeyFuncClass() {}
  virtual void foo();
};
constexpr KeyFuncClass keyFuncClassVar;
// G32-DAG: @_ZTV12KeyFuncClass = external dllimport unnamed_addr constant [3 x i8*]

struct __declspec(dllimport) X : public virtual W {};
USECLASS(X)
// vbtable:
// MO1-DAG: @"\01??_8X@@7B@" = available_externally dllimport unnamed_addr constant [2 x i32] [i32 0, i32 4]

struct __declspec(dllimport) Y {
  int x;
};

struct __declspec(dllimport) Z { virtual ~Z() {} };
USECLASS(Z)
// User-defined dtor:
// MO1-DAG: define available_externally dllimport x86_thiscallcc void @"\01??1Z@@UAE@XZ"

namespace DontUseDtorAlias {
  struct __declspec(dllimport) A { ~A(); };
  struct __declspec(dllimport) B : A { ~B(); };
  inline A::~A() { }
  inline B::~B() { }
  // Emit a real definition of B's constructor; don't alias it to A's.
  // MO1-DAG: available_externally dllimport x86_thiscallcc void @"\01??1B@DontUseDtorAlias@@QAE@XZ"
  USECLASS(B)
}

namespace Vtordisp {
  // Don't dllimport the vtordisp.
  // MO1-DAG: define linkonce_odr x86_thiscallcc void @"\01?f@?$C@D@Vtordisp@@$4PPPPPPPM@A@AEXXZ"

  class Base {
    virtual void f() {}
  };
  template <typename T>
  class __declspec(dllimport) C : virtual public Base {
  public:
    C() {}
    virtual void f() {}
  };
  template class C<char>;
}

namespace ClassTemplateStaticDef {
  // Regular template static field:
  template <typename T> struct __declspec(dllimport) S {
    static int x;
  };
  template <typename T> int S<T>::x;
  // MSC-DAG: @"\01?x@?$S@H@ClassTemplateStaticDef@@2HA" = available_externally dllimport global i32 0
  int f() { return S<int>::x; }

  // Partial class template specialization static field:
  template <typename A> struct T;
  template <typename A> struct __declspec(dllimport) T<A*> {
    static int x;
  };
  template <typename A> int T<A*>::x;
  // GNU-DAG: @_ZN22ClassTemplateStaticDef1TIPvE1xE = available_externally dllimport global i32 0
  int g() { return T<void*>::x; }
}

namespace PR19933 {
// Don't dynamically initialize dllimport vars.
// MSC2-NOT: @llvm.global_ctors
// GNU2-NOT: @llvm.global_ctors

  struct NonPOD { NonPOD(); };
  template <typename T> struct A { static NonPOD x; };
  template <typename T> NonPOD A<T>::x;
  template struct __declspec(dllimport) A<int>;
  // MSC-DAG: @"\01?x@?$A@H@PR19933@@2UNonPOD@2@A" = available_externally dllimport global %"struct.PR19933::NonPOD" zeroinitializer

  int f();
  template <typename T> struct B { static int x; };
  template <typename T> int B<T>::x = f();
  template struct __declspec(dllimport) B<int>;
  // MSC-DAG: @"\01?x@?$B@H@PR19933@@2HA" = available_externally dllimport global i32 0

  constexpr int g() { return 42; }
  template <typename T> struct C { static int x; };
  template <typename T> int C<T>::x = g();
  template struct __declspec(dllimport) C<int>;
  // MSC-DAG: @"\01?x@?$C@H@PR19933@@2HA" = available_externally dllimport global i32 42

  template <int I> struct D { static int x, y; };
  template <int I> int D<I>::x = I + 1;
  template <int I> int D<I>::y = I + f();
  template struct __declspec(dllimport) D<42>;
  // MSC-DAG: @"\01?x@?$D@$0CK@@PR19933@@2HA" = available_externally dllimport global i32 43
  // MSC-DAG: @"\01?y@?$D@$0CK@@PR19933@@2HA" = available_externally dllimport global i32 0
}

namespace PR21355 {
  struct __declspec(dllimport) S {
    virtual ~S();
  };
  S::~S() {}

  // S::~S is a key function, so we would ordinarily emit a strong definition for
  // the vtable. However, S is imported, so the vtable should be too.

  // GNU-DAG: @_ZTVN7PR213551SE = available_externally dllimport unnamed_addr constant [4 x i8*]
}

namespace PR21366 {
  struct __declspec(dllimport) S {
    void outOfLineMethod();
    void inlineMethod() {}
    inline void anotherInlineMethod();
    void outOfClassInlineMethod();
  };
  void S::anotherInlineMethod() {}
  inline void S::outOfClassInlineMethod() {}
}

// MS ignores DLL attributes on partial specializations.
template <typename T> struct PartiallySpecializedClassTemplate {};
template <typename T> struct __declspec(dllimport) PartiallySpecializedClassTemplate<T*> { void f(); };
USEMEMFUNC(PartiallySpecializedClassTemplate<void*>, f);
// M32-DAG: declare x86_thiscallcc void @"\01?f@?$PartiallySpecializedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN33PartiallySpecializedClassTemplateIPvE1fEv

// Attributes on explicit specializations are honored.
template <typename T> struct ExplicitlySpecializedClassTemplate {};
template <> struct __declspec(dllimport) ExplicitlySpecializedClassTemplate<void*> { void f(); };
USEMEMFUNC(ExplicitlySpecializedClassTemplate<void*>, f);
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?f@?$ExplicitlySpecializedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN34ExplicitlySpecializedClassTemplateIPvE1fEv

// MS inherits DLL attributes to partial specializations.
template <typename T> struct __declspec(dllimport) PartiallySpecializedImportedClassTemplate {};
template <typename T> struct PartiallySpecializedImportedClassTemplate<T*> { void f() {} };
USEMEMFUNC(PartiallySpecializedImportedClassTemplate<void*>, f);
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?f@?$PartiallySpecializedImportedClassTemplate@PAX@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN41PartiallySpecializedImportedClassTemplateIPvE1fEv

// Attributes on the instantiation take precedence over attributes on the template.
template <typename T> struct __declspec(dllexport) ExplicitlyInstantiatedWithDifferentAttr { void f() {} };
template struct __declspec(dllimport) ExplicitlyInstantiatedWithDifferentAttr<int>;
USEMEMFUNC(ExplicitlyInstantiatedWithDifferentAttr<int>, f);
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?f@?$ExplicitlyInstantiatedWithDifferentAttr@H@@QAEXXZ"


//===----------------------------------------------------------------------===//
// Classes with template base classes
//===----------------------------------------------------------------------===//

template <typename T> struct ClassTemplate { void func() {} };
template <typename T> struct __declspec(dllexport) ExportedClassTemplate { void func() {} };
template <typename T> struct __declspec(dllimport) ImportedClassTemplate { void func(); };

template <typename T> struct ExplicitlySpecializedTemplate { void func() {} };
template <> struct ExplicitlySpecializedTemplate<int> { void func() {} };
template <typename T> struct ExplicitlyExportSpecializedTemplate { void func() {} };
template <> struct __declspec(dllexport) ExplicitlyExportSpecializedTemplate<int> { void func() {} };
template <typename T> struct ExplicitlyImportSpecializedTemplate { void func() {} };
template <> struct __declspec(dllimport) ExplicitlyImportSpecializedTemplate<int> { void func(); };

template <typename T> struct ExplicitlyInstantiatedTemplate { void func() {} };
template struct ExplicitlyInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyExportInstantiatedTemplate { void func() {} };
template struct __declspec(dllexport) ExplicitlyExportInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyImportInstantiatedTemplate { void func(); };
template struct __declspec(dllimport) ExplicitlyImportInstantiatedTemplate<int>;


// MS: ClassTemplate<int> gets imported.
struct __declspec(dllimport) DerivedFromTemplate : public ClassTemplate<int> {};
USEMEMFUNC(ClassTemplate<int>, func)
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?func@?$ClassTemplate@H@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIiE4funcEv

// ImportedTemplate is explicitly imported.
struct __declspec(dllimport) DerivedFromImportedTemplate : public ImportedClassTemplate<int> {};
USEMEMFUNC(ImportedClassTemplate<int>, func)
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?func@?$ImportedClassTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN21ImportedClassTemplateIiE4funcEv

// ExportedTemplate is explicitly exported.
struct __declspec(dllimport) DerivedFromExportedTemplate : public ExportedClassTemplate<int> {};
USEMEMFUNC(ExportedClassTemplate<int>, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExportedClassTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN21ExportedClassTemplateIiE4funcEv

// Base class already instantiated without attribute.
struct DerivedFromTemplateD : public ClassTemplate<double> {};
struct __declspec(dllimport) DerivedFromTemplateD2 : public ClassTemplate<double> {};
USEMEMFUNC(ClassTemplate<double>, func)
// M32-DAG: define linkonce_odr x86_thiscallcc void @"\01?func@?$ClassTemplate@N@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIdE4funcEv

// MS: Base class already instantiated with dfferent attribute.
struct __declspec(dllexport) DerivedFromTemplateB : public ClassTemplate<bool> {};
struct __declspec(dllimport) DerivedFromTemplateB2 : public ClassTemplate<bool> {};
USEMEMFUNC(ClassTemplate<bool>, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ClassTemplate@_N@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN13ClassTemplateIbE4funcEv

// Base class already specialized without dll attribute.
struct __declspec(dllimport) DerivedFromExplicitlySpecializedTemplate : public ExplicitlySpecializedTemplate<int> {};
USEMEMFUNC(ExplicitlySpecializedTemplate<int>, func)
// M32-DAG: define linkonce_odr x86_thiscallcc void @"\01?func@?$ExplicitlySpecializedTemplate@H@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN29ExplicitlySpecializedTemplateIiE4funcEv

// Base class alredy specialized with export attribute.
struct __declspec(dllimport) DerivedFromExplicitlyExportSpecializedTemplate : public ExplicitlyExportSpecializedTemplate<int> {};
USEMEMFUNC(ExplicitlyExportSpecializedTemplate<int>, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExplicitlyExportSpecializedTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN35ExplicitlyExportSpecializedTemplateIiE4funcEv

// Base class already specialized with import attribute.
struct __declspec(dllimport) DerivedFromExplicitlyImportSpecializedTemplate : public ExplicitlyImportSpecializedTemplate<int> {};
USEMEMFUNC(ExplicitlyImportSpecializedTemplate<int>, func)
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?func@?$ExplicitlyImportSpecializedTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN35ExplicitlyImportSpecializedTemplateIiE4funcEv

// Base class already instantiated without dll attribute.
struct __declspec(dllimport) DerivedFromExplicitlyInstantiatedTemplate : public ExplicitlyInstantiatedTemplate<int> {};
USEMEMFUNC(ExplicitlyInstantiatedTemplate<int>, func)
// M32-DAG: define weak_odr x86_thiscallcc void @"\01?func@?$ExplicitlyInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr x86_thiscallcc void @_ZN30ExplicitlyInstantiatedTemplateIiE4funcEv

// Base class already instantiated with export attribute.
struct __declspec(dllimport) DerivedFromExplicitlyExportInstantiatedTemplate : public ExplicitlyExportInstantiatedTemplate<int> {};
USEMEMFUNC(ExplicitlyExportInstantiatedTemplate<int>, func)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?func@?$ExplicitlyExportInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN36ExplicitlyExportInstantiatedTemplateIiE4funcEv

// Base class already instantiated with import attribute.
struct __declspec(dllimport) DerivedFromExplicitlyImportInstantiatedTemplate : public ExplicitlyImportInstantiatedTemplate<int> {};
USEMEMFUNC(ExplicitlyImportInstantiatedTemplate<int>, func)
// M32-DAG: declare dllimport x86_thiscallcc void @"\01?func@?$ExplicitlyImportInstantiatedTemplate@H@@QAEXXZ"
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN36ExplicitlyImportInstantiatedTemplateIiE4funcEv

// MS: A dll attribute propagates through multiple levels of instantiation.
template <typename T> struct TopClass { void func() {} };
template <typename T> struct MiddleClass : public TopClass<T> { };
struct __declspec(dllimport) BottomClass : public MiddleClass<int> { };
USEMEMFUNC(TopClass<int>, func)
// M32-DAG: {{declare|define available_externally}} dllimport x86_thiscallcc void @"\01?func@?$TopClass@H@@QAEXXZ"
// G32-DAG: define linkonce_odr x86_thiscallcc void @_ZN8TopClassIiE4funcEv
