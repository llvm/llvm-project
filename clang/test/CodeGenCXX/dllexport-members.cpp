// RUN: %clang_cc1 -triple i686-windows-msvc   -fms-compatibility -fms-compatibility-version=18 -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=MSC --check-prefix=M32 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fms-compatibility-version=18 -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=MSC --check-prefix=M64 %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -fms-compatibility -fms-compatibility-version=19 -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=M32VS2015 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-compatibility -fms-compatibility-version=19 -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=M64VS2015 %s
// RUN: %clang_cc1 -triple i686-windows-gnu                       -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu                     -emit-llvm -std=c++1y -O0 -o - %s | FileCheck --check-prefix=GNU --check-prefix=G64 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Exported {};
struct ExplicitDecl_Exported {};
struct ExplicitInst_Exported {};
struct ExplicitSpec_Exported {};
struct ExplicitSpec_Def_Exported {};
struct ExplicitSpec_InlineDef_Exported {};
struct ExplicitSpec_NotExported {};

extern "C" void* malloc(__SIZE_TYPE__ size);
extern "C" void free(void* p);


//===----------------------------------------------------------------------===//
// Class members
//===----------------------------------------------------------------------===//

// Export individual members of a class.
struct ExportMembers {
  struct Nested;

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?normalDef@ExportMembers@@QAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define          dllexport                void @"\01?normalDef@ExportMembers@@QEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInclass@ExportMembers@@QAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInclass@ExportMembers@@QEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInlineDef@ExportMembers@@QAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInlineDef@ExportMembers@@QEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInlineDecl@ExportMembers@@QAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInlineDecl@ExportMembers@@QEAAXXZ"(%struct.ExportMembers* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers9normalDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers9normalDefEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers13normalInclassEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers13normalInclassEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers15normalInlineDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers15normalInlineDefEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers16normalInlineDeclEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers16normalInlineDeclEv(%struct.ExportMembers* %this)
  // M32-DAG: define linkonce_odr       x86_thiscallcc void @"\01?referencedNonExportedInClass@ExportMembers@@QAEXXZ"
  __declspec(dllexport)                void normalDef();
  __declspec(dllexport)                void normalInclass() { referencedNonExportedInClass(); }
  __declspec(dllexport)                void normalInlineDef();
  __declspec(dllexport)         inline void normalInlineDecl();
                                       void referencedNonExportedInClass() {}

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?virtualDef@ExportMembers@@UAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define          dllexport                void @"\01?virtualDef@ExportMembers@@UEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInclass@ExportMembers@@UAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInclass@ExportMembers@@UEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInlineDef@ExportMembers@@UAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInlineDef@ExportMembers@@UEAAXXZ"(%struct.ExportMembers* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInlineDecl@ExportMembers@@UAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInlineDecl@ExportMembers@@UEAAXXZ"(%struct.ExportMembers* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers10virtualDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers10virtualDefEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers14virtualInclassEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers14virtualInclassEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers16virtualInlineDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers16virtualInlineDefEv(%struct.ExportMembers* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers17virtualInlineDeclEv(%struct.ExportMembers* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers17virtualInlineDeclEv(%struct.ExportMembers* %this)
  __declspec(dllexport) virtual        void virtualDef();
  __declspec(dllexport) virtual        void virtualInclass() {}
  __declspec(dllexport) virtual        void virtualInlineDef();
  __declspec(dllexport) virtual inline void virtualInlineDecl();

  // MSC-DAG: define          dllexport                void @"\01?staticDef@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInclass@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInlineDef@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInlineDecl@ExportMembers@@SAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers9staticDefEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers13staticInclassEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers15staticInlineDefEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers16staticInlineDeclEv()
  __declspec(dllexport) static         void staticDef();
  __declspec(dllexport) static         void staticInclass() {}
  __declspec(dllexport) static         void staticInlineDef();
  __declspec(dllexport) static  inline void staticInlineDecl();

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?protectedDef@ExportMembers@@IAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define          dllexport                void @"\01?protectedDef@ExportMembers@@IEAAXXZ"(%struct.ExportMembers* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers12protectedDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers12protectedDefEv(%struct.ExportMembers* %this)
  // MSC-DAG: define          dllexport                void @"\01?protectedStaticDef@ExportMembers@@KAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers18protectedStaticDefEv()
protected:
  __declspec(dllexport)                void protectedDef();
  __declspec(dllexport) static         void protectedStaticDef();

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?privateDef@ExportMembers@@AAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define          dllexport                void @"\01?privateDef@ExportMembers@@AEAAXXZ"(%struct.ExportMembers* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers10privateDefEv(%struct.ExportMembers* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers10privateDefEv(%struct.ExportMembers* %this)
  // MSC-DAG: define          dllexport                void @"\01?privateStaticDef@ExportMembers@@CAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers16privateStaticDefEv()
private:
  __declspec(dllexport)                void privateDef();
  __declspec(dllexport) static         void privateStaticDef();

  // M32-DAG: define                    x86_thiscallcc void @"\01?ignored@ExportMembers@@QAEXXZ"(%struct.ExportMembers* %this)
  // M64-DAG: define                                   void @"\01?ignored@ExportMembers@@QEAAXXZ"(%struct.ExportMembers* %this)
  // G32-DAG: define                    x86_thiscallcc void @_ZN13ExportMembers7ignoredEv(%struct.ExportMembers* %this)
  // G64-DAG: define                                   void @_ZN13ExportMembers7ignoredEv(%struct.ExportMembers* %this)
public:
  void ignored();

  // MSC-DAG: @"\01?StaticField@ExportMembers@@2HA"               = dllexport global i32 1, align 4
  // MSC-DAG: @"\01?StaticConstField@ExportMembers@@2HB"          = dllexport constant i32 1, align 4
  // MSC-DAG: @"\01?StaticConstFieldEqualInit@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?StaticConstFieldBraceInit@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?StaticConstFieldRefNotDef@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?ConstexprField@ExportMembers@@2HB"            = weak_odr dllexport constant i32 1, comdat, align 4
  // GNU-DAG: @_ZN13ExportMembers11StaticFieldE                   = dllexport global i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers16StaticConstFieldE              = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers25StaticConstFieldEqualInitE     = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers25StaticConstFieldBraceInitE     = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers14ConstexprFieldE                = dllexport constant i32 1, align 4
  __declspec(dllexport) static         int  StaticField;
  __declspec(dllexport) static  const  int  StaticConstField;
  __declspec(dllexport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllexport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllexport) static  const  int  StaticConstFieldRefNotDef = 1;
  __declspec(dllexport) constexpr static int ConstexprField = 1;
};

       void ExportMembers::normalDef() {}
inline void ExportMembers::normalInlineDef() {}
       void ExportMembers::normalInlineDecl() {}
       void ExportMembers::virtualDef() {}
inline void ExportMembers::virtualInlineDef() {}
       void ExportMembers::virtualInlineDecl() {}
       void ExportMembers::staticDef() {}
inline void ExportMembers::staticInlineDef() {}
       void ExportMembers::staticInlineDecl() {}
       void ExportMembers::ignored() {}
       void ExportMembers::protectedDef() {}
       void ExportMembers::protectedStaticDef() {}
       void ExportMembers::privateDef() {}
       void ExportMembers::privateStaticDef() {}

       int  ExportMembers::StaticField = 1;
const  int  ExportMembers::StaticConstField = 1;
const  int  ExportMembers::StaticConstFieldEqualInit;
const  int  ExportMembers::StaticConstFieldBraceInit;
int foo() { return ExportMembers::StaticConstFieldRefNotDef; }
constexpr int ExportMembers::ConstexprField;


// Export individual members of a nested class.
struct ExportMembers::Nested {
  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?normalDef@Nested@ExportMembers@@QAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define          dllexport                void @"\01?normalDef@Nested@ExportMembers@@QEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInclass@Nested@ExportMembers@@QAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInclass@Nested@ExportMembers@@QEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInlineDef@Nested@ExportMembers@@QAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInlineDef@Nested@ExportMembers@@QEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?normalInlineDecl@Nested@ExportMembers@@QAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?normalInlineDecl@Nested@ExportMembers@@QEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested9normalDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers6Nested9normalDefEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested13normalInclassEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested13normalInclassEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested15normalInlineDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested15normalInlineDefEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested16normalInlineDeclEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested16normalInlineDeclEv(%"struct.ExportMembers::Nested"* %this)
  __declspec(dllexport)                void normalDef();
  __declspec(dllexport)                void normalInclass() {}
  __declspec(dllexport)                void normalInlineDef();
  __declspec(dllexport)         inline void normalInlineDecl();

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?virtualDef@Nested@ExportMembers@@UAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define          dllexport                void @"\01?virtualDef@Nested@ExportMembers@@UEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInclass@Nested@ExportMembers@@UAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInclass@Nested@ExportMembers@@UEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInlineDef@Nested@ExportMembers@@UAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInlineDef@Nested@ExportMembers@@UEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?virtualInlineDecl@Nested@ExportMembers@@UAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01?virtualInlineDecl@Nested@ExportMembers@@UEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested10virtualDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers6Nested10virtualDefEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested14virtualInclassEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested14virtualInclassEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested16virtualInlineDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested16virtualInlineDefEv(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested17virtualInlineDeclEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested17virtualInlineDeclEv(%"struct.ExportMembers::Nested"* %this)
  __declspec(dllexport) virtual        void virtualDef();
  __declspec(dllexport) virtual        void virtualInclass() {}
  __declspec(dllexport) virtual        void virtualInlineDef();
  __declspec(dllexport) virtual inline void virtualInlineDecl();

  // MSC-DAG: define          dllexport                void @"\01?staticDef@Nested@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInclass@Nested@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInlineDef@Nested@ExportMembers@@SAXXZ"()
  // MSC-DAG: define weak_odr dllexport                void @"\01?staticInlineDecl@Nested@ExportMembers@@SAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers6Nested9staticDefEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested13staticInclassEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested15staticInlineDefEv()
  // GNU-DAG: define weak_odr dllexport                void @_ZN13ExportMembers6Nested16staticInlineDeclEv()
  __declspec(dllexport) static         void staticDef();
  __declspec(dllexport) static         void staticInclass() {}
  __declspec(dllexport) static         void staticInlineDef();
  __declspec(dllexport) static  inline void staticInlineDecl();

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?protectedDef@Nested@ExportMembers@@IAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define          dllexport                void @"\01?protectedDef@Nested@ExportMembers@@IEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested12protectedDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers6Nested12protectedDefEv(%"struct.ExportMembers::Nested"* %this)
  // MSC-DAG: define          dllexport                void @"\01?protectedStaticDef@Nested@ExportMembers@@KAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers6Nested18protectedStaticDefEv()
protected:
  __declspec(dllexport)                void protectedDef();
  __declspec(dllexport) static         void protectedStaticDef();

  // M32-DAG: define          dllexport x86_thiscallcc void @"\01?privateDef@Nested@ExportMembers@@AAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define          dllexport                void @"\01?privateDef@Nested@ExportMembers@@AEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define          dllexport x86_thiscallcc void @_ZN13ExportMembers6Nested10privateDefEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define          dllexport                void @_ZN13ExportMembers6Nested10privateDefEv(%"struct.ExportMembers::Nested"* %this)
  // MSC-DAG: define          dllexport                void @"\01?privateStaticDef@Nested@ExportMembers@@CAXXZ"()
  // GNU-DAG: define          dllexport                void @_ZN13ExportMembers6Nested16privateStaticDefEv()
private:
  __declspec(dllexport)                void privateDef();
  __declspec(dllexport) static         void privateStaticDef();

  // M32-DAG: define                    x86_thiscallcc void @"\01?ignored@Nested@ExportMembers@@QAEXXZ"(%"struct.ExportMembers::Nested"* %this)
  // M64-DAG: define                                   void @"\01?ignored@Nested@ExportMembers@@QEAAXXZ"(%"struct.ExportMembers::Nested"* %this)
  // G32-DAG: define                    x86_thiscallcc void @_ZN13ExportMembers6Nested7ignoredEv(%"struct.ExportMembers::Nested"* %this)
  // G64-DAG: define                                   void @_ZN13ExportMembers6Nested7ignoredEv(%"struct.ExportMembers::Nested"* %this)
public:
  void ignored();

  // MSC-DAG: @"\01?StaticField@Nested@ExportMembers@@2HA"               = dllexport global i32 1, align 4
  // MSC-DAG: @"\01?StaticConstField@Nested@ExportMembers@@2HB"          = dllexport constant i32 1, align 4
  // MSC-DAG: @"\01?StaticConstFieldEqualInit@Nested@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?StaticConstFieldBraceInit@Nested@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?StaticConstFieldRefNotDef@Nested@ExportMembers@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
  // MSC-DAG: @"\01?ConstexprField@Nested@ExportMembers@@2HB"            = weak_odr dllexport constant i32 1, comdat, align 4
  // GNU-DAG: @_ZN13ExportMembers6Nested11StaticFieldE                   = dllexport global i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers6Nested16StaticConstFieldE              = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers6Nested25StaticConstFieldEqualInitE     = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers6Nested25StaticConstFieldBraceInitE     = dllexport constant i32 1, align 4
  // GNU-DAG: @_ZN13ExportMembers6Nested14ConstexprFieldE                = dllexport constant i32 1, align 4
  __declspec(dllexport) static         int  StaticField;
  __declspec(dllexport) static  const  int  StaticConstField;
  __declspec(dllexport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllexport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllexport) static  const  int  StaticConstFieldRefNotDef = 1;
  __declspec(dllexport) constexpr static int ConstexprField = 1;
};

       void ExportMembers::Nested::normalDef() {}
inline void ExportMembers::Nested::normalInlineDef() {}
       void ExportMembers::Nested::normalInlineDecl() {}
       void ExportMembers::Nested::virtualDef() {}
inline void ExportMembers::Nested::virtualInlineDef() {}
       void ExportMembers::Nested::virtualInlineDecl() {}
       void ExportMembers::Nested::staticDef() {}
inline void ExportMembers::Nested::staticInlineDef() {}
       void ExportMembers::Nested::staticInlineDecl() {}
       void ExportMembers::Nested::ignored() {}
       void ExportMembers::Nested::protectedDef() {}
       void ExportMembers::Nested::protectedStaticDef() {}
       void ExportMembers::Nested::privateDef() {}
       void ExportMembers::Nested::privateStaticDef() {}

       int  ExportMembers::Nested::StaticField = 1;
const  int  ExportMembers::Nested::StaticConstField = 1;
const  int  ExportMembers::Nested::StaticConstFieldEqualInit;
const  int  ExportMembers::Nested::StaticConstFieldBraceInit;
int fooNested() { return ExportMembers::Nested::StaticConstFieldRefNotDef; }
constexpr int ExportMembers::Nested::ConstexprField;


// Export special member functions.
struct ExportSpecials {
  // M32-DAG: define dllexport x86_thiscallcc %struct.ExportSpecials* @"\01??0ExportSpecials@@QAE@XZ"(%struct.ExportSpecials* returned %this)
  // M64-DAG: define dllexport                %struct.ExportSpecials* @"\01??0ExportSpecials@@QEAA@XZ"(%struct.ExportSpecials* returned %this)
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC1Ev(%struct.ExportSpecials* %this)
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC1Ev(%struct.ExportSpecials* %this)
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC2Ev(%struct.ExportSpecials* %this)
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC2Ev(%struct.ExportSpecials* %this)
  __declspec(dllexport) ExportSpecials();

  // M32-DAG: define dllexport x86_thiscallcc void @"\01??1ExportSpecials@@QAE@XZ"(%struct.ExportSpecials* %this)
  // M64-DAG: define dllexport                void @"\01??1ExportSpecials@@QEAA@XZ"(%struct.ExportSpecials* %this)
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsD1Ev(%struct.ExportSpecials* %this)
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsD1Ev(%struct.ExportSpecials* %this)
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsD2Ev(%struct.ExportSpecials* %this)
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsD2Ev(%struct.ExportSpecials* %this)
  __declspec(dllexport) ~ExportSpecials();

  // M32-DAG: define dllexport x86_thiscallcc %struct.ExportSpecials* @"\01??0ExportSpecials@@QAE@ABU0@@Z"(%struct.ExportSpecials* returned %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // M64-DAG: define dllexport                %struct.ExportSpecials* @"\01??0ExportSpecials@@QEAA@AEBU0@@Z"(%struct.ExportSpecials* returned %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC1ERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC1ERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC2ERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC2ERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  __declspec(dllexport) ExportSpecials(const ExportSpecials&);

  // M32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @"\01??4ExportSpecials@@QAEAAU0@ABU0@@Z"(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // M64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @"\01??4ExportSpecials@@QEAAAEAU0@AEBU0@@Z"(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @_ZN14ExportSpecialsaSERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @_ZN14ExportSpecialsaSERKS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  __declspec(dllexport) ExportSpecials& operator=(const ExportSpecials&);

  // M32-DAG: define dllexport x86_thiscallcc %struct.ExportSpecials* @"\01??0ExportSpecials@@QAE@$$QAU0@@Z"(%struct.ExportSpecials* returned %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // M64-DAG: define dllexport                %struct.ExportSpecials* @"\01??0ExportSpecials@@QEAA@$$QEAU0@@Z"(%struct.ExportSpecials* returned %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC1EOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC1EOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc void @_ZN14ExportSpecialsC2EOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                void @_ZN14ExportSpecialsC2EOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  __declspec(dllexport) ExportSpecials(ExportSpecials&&);

  // M32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @"\01??4ExportSpecials@@QAEAAU0@$$QAU0@@Z"(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // M64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @"\01??4ExportSpecials@@QEAAAEAU0@$$QEAU0@@Z"(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @_ZN14ExportSpecialsaSEOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  // G64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportSpecials* @_ZN14ExportSpecialsaSEOS_(%struct.ExportSpecials* %this, %struct.ExportSpecials* dereferenceable({{[0-9]+}}))
  __declspec(dllexport) ExportSpecials& operator=(ExportSpecials&&);
};
ExportSpecials::ExportSpecials() {}
ExportSpecials::~ExportSpecials() {}
ExportSpecials::ExportSpecials(const ExportSpecials&) {}
ExportSpecials& ExportSpecials::operator=(const ExportSpecials&) { return *this; }
ExportSpecials::ExportSpecials(ExportSpecials&&) {}
ExportSpecials& ExportSpecials::operator=(ExportSpecials&&) { return *this; }


// Export class with inline special member functions.
struct ExportInlineSpecials {
  // M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QAE@XZ"(%struct.ExportInlineSpecials* returned %this)
  // M64-DAG: define weak_odr dllexport                %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QEAA@XZ"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN20ExportInlineSpecialsC1Ev(
  // G64-DAG: define weak_odr dllexport                void @_ZN20ExportInlineSpecialsC1Ev(
  __declspec(dllexport) ExportInlineSpecials() {}

  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??1ExportInlineSpecials@@QAE@XZ"(
  // M64-DAG: define weak_odr dllexport                void @"\01??1ExportInlineSpecials@@QEAA@XZ"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN20ExportInlineSpecialsD1Ev(
  // G64-DAG: define weak_odr dllexport                void @_ZN20ExportInlineSpecialsD1Ev(
  __declspec(dllexport) ~ExportInlineSpecials() {}

  // M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QAE@ABU0@@Z"(
  // M64-DAG: define weak_odr dllexport                %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QEAA@AEBU0@@Z"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN20ExportInlineSpecialsC1ERKS_(
  // G64-DAG: define weak_odr dllexport                void @_ZN20ExportInlineSpecialsC1ERKS_(
  __declspec(dllexport) inline ExportInlineSpecials(const ExportInlineSpecials&);

  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @"\01??4ExportInlineSpecials@@QAEAAU0@ABU0@@Z"(
  // M64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @"\01??4ExportInlineSpecials@@QEAAAEAU0@AEBU0@@Z"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @_ZN20ExportInlineSpecialsaSERKS_(
  // G64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @_ZN20ExportInlineSpecialsaSERKS_(
  __declspec(dllexport) ExportInlineSpecials& operator=(const ExportInlineSpecials&);

  // M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QAE@$$QAU0@@Z"(
  // M64-DAG: define weak_odr dllexport                %struct.ExportInlineSpecials* @"\01??0ExportInlineSpecials@@QEAA@$$QEAU0@@Z"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN20ExportInlineSpecialsC1EOS_(
  // G64-DAG: define weak_odr dllexport                void @_ZN20ExportInlineSpecialsC1EOS_(
  __declspec(dllexport) ExportInlineSpecials(ExportInlineSpecials&&) {}

  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @"\01??4ExportInlineSpecials@@QAEAAU0@$$QAU0@@Z"(
  // M64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @"\01??4ExportInlineSpecials@@QEAAAEAU0@$$QEAU0@@Z"(
  // G32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @_ZN20ExportInlineSpecialsaSEOS_(
  // G64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportInlineSpecials* @_ZN20ExportInlineSpecialsaSEOS_(
  __declspec(dllexport) ExportInlineSpecials& operator=(ExportInlineSpecials&&) { return *this; }
};
ExportInlineSpecials::ExportInlineSpecials(const ExportInlineSpecials&) {}
inline ExportInlineSpecials& ExportInlineSpecials::operator=(const ExportInlineSpecials&) { return *this; }


// Export defaulted member function definitions.
struct ExportDefaultedDefs {
  __declspec(dllexport) ExportDefaultedDefs();
  __declspec(dllexport) ~ExportDefaultedDefs();
  __declspec(dllexport) inline ExportDefaultedDefs(const ExportDefaultedDefs&);
  __declspec(dllexport) ExportDefaultedDefs& operator=(const ExportDefaultedDefs&);
  __declspec(dllexport) ExportDefaultedDefs(ExportDefaultedDefs&&);
  __declspec(dllexport) ExportDefaultedDefs& operator=(ExportDefaultedDefs&&);
};

// M32-DAG: define dllexport x86_thiscallcc %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QAE@XZ"(%struct.ExportDefaultedDefs* returned %this)
// M64-DAG: define dllexport                %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QEAA@XZ"(%struct.ExportDefaultedDefs* returned %this)
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC1Ev(%struct.ExportDefaultedDefs* %this)
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsC1Ev(%struct.ExportDefaultedDefs* %this)
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC2Ev(%struct.ExportDefaultedDefs* %this)
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsC2Ev(%struct.ExportDefaultedDefs* %this)
__declspec(dllexport) ExportDefaultedDefs::ExportDefaultedDefs() = default;

// M32-DAG: define dllexport x86_thiscallcc void @"\01??1ExportDefaultedDefs@@QAE@XZ"(%struct.ExportDefaultedDefs* %this)
// M64-DAG: define dllexport                void @"\01??1ExportDefaultedDefs@@QEAA@XZ"(%struct.ExportDefaultedDefs* %this)
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsD1Ev(%struct.ExportDefaultedDefs* %this)
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsD1Ev(%struct.ExportDefaultedDefs* %this)
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsD2Ev(%struct.ExportDefaultedDefs* %this)
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsD2Ev(%struct.ExportDefaultedDefs* %this)
ExportDefaultedDefs::~ExportDefaultedDefs() = default;

// M32-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QAE@ABU0@@Z"(%struct.ExportDefaultedDefs* returned %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// M64-DAG: define weak_odr dllexport                %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QEAA@AEBU0@@Z"(%struct.ExportDefaultedDefs* returned %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC1ERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define weak_odr dllexport                void @_ZN19ExportDefaultedDefsC1ERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC2ERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define weak_odr dllexport                void @_ZN19ExportDefaultedDefsC2ERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
__declspec(dllexport) ExportDefaultedDefs::ExportDefaultedDefs(const ExportDefaultedDefs&) = default;

// M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @"\01??4ExportDefaultedDefs@@QAEAAU0@ABU0@@Z"(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// M64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @"\01??4ExportDefaultedDefs@@QEAAAEAU0@AEBU0@@Z"(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @_ZN19ExportDefaultedDefsaSERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @_ZN19ExportDefaultedDefsaSERKS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
inline ExportDefaultedDefs& ExportDefaultedDefs::operator=(const ExportDefaultedDefs&) = default;

// M32-DAG: define dllexport x86_thiscallcc %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QAE@$$QAU0@@Z"(%struct.ExportDefaultedDefs* returned %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// M64-DAG: define dllexport                %struct.ExportDefaultedDefs* @"\01??0ExportDefaultedDefs@@QEAA@$$QEAU0@@Z"(%struct.ExportDefaultedDefs* returned %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC1EOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsC1EOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define dllexport x86_thiscallcc void @_ZN19ExportDefaultedDefsC2EOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define dllexport                void @_ZN19ExportDefaultedDefsC2EOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
__declspec(dllexport) ExportDefaultedDefs::ExportDefaultedDefs(ExportDefaultedDefs&&) = default;

// M32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @"\01??4ExportDefaultedDefs@@QAEAAU0@$$QAU0@@Z"(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// M64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @"\01??4ExportDefaultedDefs@@QEAAAEAU0@$$QEAU0@@Z"(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G32-DAG: define dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @_ZN19ExportDefaultedDefsaSEOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
// G64-DAG: define dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedDefs* @_ZN19ExportDefaultedDefsaSEOS_(%struct.ExportDefaultedDefs* %this, %struct.ExportDefaultedDefs* dereferenceable({{[0-9]+}}))
ExportDefaultedDefs& ExportDefaultedDefs::operator=(ExportDefaultedDefs&&) = default;


// Export defaulted member function definitions declared inside class.
struct ExportDefaultedInclassDefs {
  __declspec(dllexport) ExportDefaultedInclassDefs() = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M64VS2013-DAG: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)
  // M64VS2015-NOT: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* returned %this)

  __declspec(dllexport) ~ExportDefaultedInclassDefs() = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??1ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M64VS2013-DAG: define weak_odr dllexport                void @"\01??1ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc void @"\01??1ExportDefaultedInclassDefs@@QAE@XZ"(%struct.ExportDefaultedInclassDefs* %this)
  // M64VS2015-NOT: define weak_odr dllexport                void @"\01??1ExportDefaultedInclassDefs@@QEAA@XZ"(%struct.ExportDefaultedInclassDefs* %this)

  __declspec(dllexport) ExportDefaultedInclassDefs(const ExportDefaultedInclassDefs&) = default;
  // M32VS2013-DAG: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64VS2013-DAG: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M32VS2015-NOT: define weak_odr dllexport x86_thiscallcc %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QAE@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64VS2015-NOT: define weak_odr dllexport                %struct.ExportDefaultedInclassDefs* @"\01??0ExportDefaultedInclassDefs@@QEAA@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* returned %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))

  __declspec(dllexport) ExportDefaultedInclassDefs& operator=(const ExportDefaultedInclassDefs&) = default;
  // M32-DAG: define weak_odr dllexport x86_thiscallcc dereferenceable({{[0-9]+}}) %struct.ExportDefaultedInclassDefs* @"\01??4ExportDefaultedInclassDefs@@QAEAAU0@ABU0@@Z"(%struct.ExportDefaultedInclassDefs* %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
  // M64-DAG: define weak_odr dllexport                dereferenceable({{[0-9]+}}) %struct.ExportDefaultedInclassDefs* @"\01??4ExportDefaultedInclassDefs@@QEAAAEAU0@AEBU0@@Z"(%struct.ExportDefaultedInclassDefs* %this, %struct.ExportDefaultedInclassDefs* dereferenceable({{[0-9]+}}))
};


// Export allocation functions.
struct ExportAlloc {
  __declspec(dllexport) void* operator new(__SIZE_TYPE__);
  __declspec(dllexport) void* operator new[](__SIZE_TYPE__);
  __declspec(dllexport) void operator delete(void*);
  __declspec(dllexport) void operator delete[](void*);
};

// M32-DAG: define dllexport i8* @"\01??2ExportAlloc@@SAPAXI@Z"(i32 %n)
// M64-DAG: define dllexport i8* @"\01??2ExportAlloc@@SAPEAX_K@Z"(i64 %n)
// G32-DAG: define dllexport i8* @_ZN11ExportAllocnwEj(i32 %n)
// G64-DAG: define dllexport i8* @_ZN11ExportAllocnwEy(i64 %n)
void* ExportAlloc::operator new(__SIZE_TYPE__ n) { return malloc(n); }

// M32-DAG: define dllexport i8* @"\01??_UExportAlloc@@SAPAXI@Z"(i32 %n)
// M64-DAG: define dllexport i8* @"\01??_UExportAlloc@@SAPEAX_K@Z"(i64 %n)
// G32-DAG: define dllexport i8* @_ZN11ExportAllocnaEj(i32 %n)
// G64-DAG: define dllexport i8* @_ZN11ExportAllocnaEy(i64 %n)
void* ExportAlloc::operator new[](__SIZE_TYPE__ n) { return malloc(n); }

// M32-DAG: define dllexport void @"\01??3ExportAlloc@@SAXPAX@Z"(i8* %p)
// M64-DAG: define dllexport void @"\01??3ExportAlloc@@SAXPEAX@Z"(i8* %p)
// G32-DAG: define dllexport void @_ZN11ExportAllocdlEPv(i8* %p)
// G64-DAG: define dllexport void @_ZN11ExportAllocdlEPv(i8* %p)
void ExportAlloc::operator delete(void* p) { free(p); }

// M32-DAG: define dllexport void @"\01??_VExportAlloc@@SAXPAX@Z"(i8* %p)
// M64-DAG: define dllexport void @"\01??_VExportAlloc@@SAXPEAX@Z"(i8* %p)
// G32-DAG: define dllexport void @_ZN11ExportAllocdaEPv(i8* %p)
// G64-DAG: define dllexport void @_ZN11ExportAllocdaEPv(i8* %p)
void ExportAlloc::operator delete[](void* p) { free(p); }


//===----------------------------------------------------------------------===//
// Class member templates
//===----------------------------------------------------------------------===//

struct MemFunTmpl {
  template<typename T>                              void normalDef() {}
  template<typename T> __declspec(dllexport)        void exportedNormal() {}
  template<typename T>                       static void staticDef() {}
  template<typename T> __declspec(dllexport) static void exportedStatic() {}
};

// Export implicit instantiation of an exported member function template.
void useMemFunTmpl() {
  // M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$exportedNormal@UImplicitInst_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
  // M64-DAG: define weak_odr dllexport                void @"\01??$exportedNormal@UImplicitInst_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
  // G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI21ImplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
  // G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedNormalI21ImplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
  MemFunTmpl().exportedNormal<ImplicitInst_Exported>();

  // MSC-DAG: define weak_odr dllexport                void @"\01??$exportedStatic@UImplicitInst_Exported@@@MemFunTmpl@@SAXXZ"()
  // GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedStaticI21ImplicitInst_ExportedEEvv()
  MemFunTmpl().exportedStatic<ImplicitInst_Exported>();
}


// Export explicit instantiation declaration of an exported member function
// template.
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$exportedNormal@UExplicitDecl_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$exportedNormal@UExplicitDecl_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI21ExplicitDecl_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedNormalI21ExplicitDecl_ExportedEEvv(%struct.MemFunTmpl* %this)
extern template void MemFunTmpl::exportedNormal<ExplicitDecl_Exported>();
       template void MemFunTmpl::exportedNormal<ExplicitDecl_Exported>();

// MSC-DAG: define weak_odr dllexport                void @"\01??$exportedStatic@UExplicitDecl_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedStaticI21ExplicitDecl_ExportedEEvv()
extern template void MemFunTmpl::exportedStatic<ExplicitDecl_Exported>();
       template void MemFunTmpl::exportedStatic<ExplicitDecl_Exported>();


// Export explicit instantiation definition of an exported member function
// template.
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$exportedNormal@UExplicitInst_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$exportedNormal@UExplicitInst_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI21ExplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedNormalI21ExplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
template void MemFunTmpl::exportedNormal<ExplicitInst_Exported>();

// MSC-DAG: define weak_odr dllexport                void @"\01??$exportedStatic@UExplicitInst_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedStaticI21ExplicitInst_ExportedEEvv()
template void MemFunTmpl::exportedStatic<ExplicitInst_Exported>();


// Export specialization of an exported member function template.
// M32-DAG: define          dllexport x86_thiscallcc void @"\01??$exportedNormal@UExplicitSpec_Def_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define          dllexport                void @"\01??$exportedNormal@UExplicitSpec_Def_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define          dllexport x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI25ExplicitSpec_Def_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define          dllexport                void @_ZN10MemFunTmpl14exportedNormalI25ExplicitSpec_Def_ExportedEEvv(%struct.MemFunTmpl* %this)
template<> __declspec(dllexport) void MemFunTmpl::exportedNormal<ExplicitSpec_Def_Exported>() {}

// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$exportedNormal@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$exportedNormal@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI31ExplicitSpec_InlineDef_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedNormalI31ExplicitSpec_InlineDef_ExportedEEvv(%struct.MemFunTmpl* %this)
template<> __declspec(dllexport) inline void MemFunTmpl::exportedNormal<ExplicitSpec_InlineDef_Exported>() {}

// MSC-DAG: define          dllexport                void @"\01??$exportedStatic@UExplicitSpec_Def_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define          dllexport                void @_ZN10MemFunTmpl14exportedStaticI25ExplicitSpec_Def_ExportedEEvv()
template<> __declspec(dllexport) void MemFunTmpl::exportedStatic<ExplicitSpec_Def_Exported>() {}

// MSC-DAG: define weak_odr dllexport                void @"\01??$exportedStatic@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl14exportedStaticI31ExplicitSpec_InlineDef_ExportedEEvv()
template<> __declspec(dllexport) inline void MemFunTmpl::exportedStatic<ExplicitSpec_InlineDef_Exported>() {}


// Not exporting specialization of an exported member function template without
// explicit dllexport.
// M32-DAG: define                    x86_thiscallcc void @"\01??$exportedNormal@UExplicitSpec_NotExported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define                                   void @"\01??$exportedNormal@UExplicitSpec_NotExported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define                    x86_thiscallcc void @_ZN10MemFunTmpl14exportedNormalI24ExplicitSpec_NotExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define                                   void @_ZN10MemFunTmpl14exportedNormalI24ExplicitSpec_NotExportedEEvv(%struct.MemFunTmpl* %this)
template<> void MemFunTmpl::exportedNormal<ExplicitSpec_NotExported>() {}

// M32-DAG: define                                   void @"\01??$exportedStatic@UExplicitSpec_NotExported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define                                   void @_ZN10MemFunTmpl14exportedStaticI24ExplicitSpec_NotExportedEEvv()
template<> void MemFunTmpl::exportedStatic<ExplicitSpec_NotExported>() {}


// Export explicit instantiation declaration of a non-exported member function
// template.
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$normalDef@UExplicitDecl_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$normalDef@UExplicitDecl_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI21ExplicitDecl_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9normalDefI21ExplicitDecl_ExportedEEvv(%struct.MemFunTmpl* %this)
extern template __declspec(dllexport) void MemFunTmpl::normalDef<ExplicitDecl_Exported>();
       template __declspec(dllexport) void MemFunTmpl::normalDef<ExplicitDecl_Exported>();

// M32-DAG: define weak_odr dllexport                void @"\01??$staticDef@UExplicitDecl_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9staticDefI21ExplicitDecl_ExportedEEvv()
extern template __declspec(dllexport) void MemFunTmpl::staticDef<ExplicitDecl_Exported>();
       template __declspec(dllexport) void MemFunTmpl::staticDef<ExplicitDecl_Exported>();


// Export explicit instantiation definition of a non-exported member function
// template.
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$normalDef@UExplicitInst_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$normalDef@UExplicitInst_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI21ExplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9normalDefI21ExplicitInst_ExportedEEvv(%struct.MemFunTmpl* %this)
template __declspec(dllexport) void MemFunTmpl::normalDef<ExplicitInst_Exported>();

// MSC-DAG: define weak_odr dllexport                void @"\01??$staticDef@UExplicitInst_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9staticDefI21ExplicitInst_ExportedEEvv()
template __declspec(dllexport) void MemFunTmpl::staticDef<ExplicitInst_Exported>();


// Export specialization of a non-exported member function template.
// M32-DAG: define          dllexport x86_thiscallcc void @"\01??$normalDef@UExplicitSpec_Def_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define          dllexport                void @"\01??$normalDef@UExplicitSpec_Def_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// M32-DAG: define weak_odr dllexport x86_thiscallcc void @"\01??$normalDef@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@QAEXXZ"(%struct.MemFunTmpl* %this)
// M64-DAG: define weak_odr dllexport                void @"\01??$normalDef@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@QEAAXXZ"(%struct.MemFunTmpl* %this)
// G32-DAG: define          dllexport x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI25ExplicitSpec_Def_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define          dllexport                void @_ZN10MemFunTmpl9normalDefI25ExplicitSpec_Def_ExportedEEvv(%struct.MemFunTmpl* %this)
// G32-DAG: define weak_odr dllexport x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI31ExplicitSpec_InlineDef_ExportedEEvv(%struct.MemFunTmpl* %this)
// G64-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9normalDefI31ExplicitSpec_InlineDef_ExportedEEvv(%struct.MemFunTmpl* %this)
template<> __declspec(dllexport) void MemFunTmpl::normalDef<ExplicitSpec_Def_Exported>() {}
template<> __declspec(dllexport) inline void MemFunTmpl::normalDef<ExplicitSpec_InlineDef_Exported>() {}

// MSC-DAG: define          dllexport                void @"\01??$staticDef@UExplicitSpec_Def_Exported@@@MemFunTmpl@@SAXXZ"()
// MSC-DAG: define weak_odr dllexport                void @"\01??$staticDef@UExplicitSpec_InlineDef_Exported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define          dllexport                void @_ZN10MemFunTmpl9staticDefI25ExplicitSpec_Def_ExportedEEvv()
// GNU-DAG: define weak_odr dllexport                void @_ZN10MemFunTmpl9staticDefI31ExplicitSpec_InlineDef_ExportedEEvv()
template<> __declspec(dllexport) void MemFunTmpl::staticDef<ExplicitSpec_Def_Exported>() {}
template<> __declspec(dllexport) inline void MemFunTmpl::staticDef<ExplicitSpec_InlineDef_Exported>() {}



struct MemVarTmpl {
  template<typename T>                       static const int StaticVar = 1;
  template<typename T> __declspec(dllexport) static const int ExportedStaticVar = 1;
};
template<typename T> const int MemVarTmpl::StaticVar;
template<typename T> const int MemVarTmpl::ExportedStaticVar;

// Export implicit instantiation of an exported member variable template.
// MSC-DAG: @"\01??$ExportedStaticVar@UImplicitInst_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ExportedStaticVarI21ImplicitInst_ExportedEE       = weak_odr dllexport constant i32 1, comdat, align 4
int useMemVarTmpl() { return MemVarTmpl::ExportedStaticVar<ImplicitInst_Exported>; }

// Export explicit instantiation declaration of an exported member variable
// template.
// MSC-DAG: @"\01??$ExportedStaticVar@UExplicitDecl_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ExportedStaticVarI21ExplicitDecl_ExportedEE       = weak_odr dllexport constant i32 1, comdat, align 4
extern template const int MemVarTmpl::ExportedStaticVar<ExplicitDecl_Exported>;
       template const int MemVarTmpl::ExportedStaticVar<ExplicitDecl_Exported>;

// Export explicit instantiation definition of an exported member variable
// template.
// MSC-DAG: @"\01??$ExportedStaticVar@UExplicitInst_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ExportedStaticVarI21ExplicitInst_ExportedEE       = weak_odr dllexport constant i32 1, comdat, align 4
template const int MemVarTmpl::ExportedStaticVar<ExplicitInst_Exported>;

// Export specialization of an exported member variable template.
// MSC-DAG: @"\01??$ExportedStaticVar@UExplicitSpec_Def_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ExportedStaticVarI25ExplicitSpec_Def_ExportedEE       = dllexport constant i32 1, align 4
template<> __declspec(dllexport) const int MemVarTmpl::ExportedStaticVar<ExplicitSpec_Def_Exported> = 1;

// Not exporting specialization of an exported member variable template without
// explicit dllexport.
// MSC-DAG: @"\01??$ExportedStaticVar@UExplicitSpec_NotExported@@@MemVarTmpl@@2HB" = weak_odr constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ExportedStaticVarI24ExplicitSpec_NotExportedEE       = constant i32 1, align 4
template<> const int MemVarTmpl::ExportedStaticVar<ExplicitSpec_NotExported> = 1;


// Export explicit instantiation declaration of a non-exported member variable
// template.
// MSC-DAG: @"\01??$StaticVar@UExplicitDecl_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl9StaticVarI21ExplicitDecl_ExportedEE        = weak_odr dllexport constant i32 1, comdat, align 4
extern template __declspec(dllexport) const int MemVarTmpl::StaticVar<ExplicitDecl_Exported>;
       template __declspec(dllexport) const int MemVarTmpl::StaticVar<ExplicitDecl_Exported>;

// Export explicit instantiation definition of a non-exported member variable
// template.
// MSC-DAG: @"\01??$StaticVar@UExplicitInst_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl9StaticVarI21ExplicitInst_ExportedEE        = weak_odr dllexport constant i32 1, comdat, align 4
template __declspec(dllexport) const int MemVarTmpl::StaticVar<ExplicitInst_Exported>;

// Export specialization of a non-exported member variable template.
// MSC-DAG: @"\01??$StaticVar@UExplicitSpec_Def_Exported@@@MemVarTmpl@@2HB" = weak_odr dllexport constant i32 1, comdat, align 4
// GNU-DAG: @_ZN10MemVarTmpl9StaticVarI25ExplicitSpec_Def_ExportedEE        = dllexport constant i32 1, align 4
template<> __declspec(dllexport) const int MemVarTmpl::StaticVar<ExplicitSpec_Def_Exported> = 1;
