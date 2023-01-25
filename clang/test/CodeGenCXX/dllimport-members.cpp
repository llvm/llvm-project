// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple i686-windows-msvc -fms-compatibility   -emit-llvm -std=c++1y -O0 -o - %s -DMSABI | FileCheck --check-prefix=MSC --check-prefix=M32 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple x86_64-windows-msvc -fms-compatibility -emit-llvm -std=c++1y -O0 -o - %s -DMSABI | FileCheck --check-prefix=MSC --check-prefix=M64 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple i686-windows-gnu                       -emit-llvm -std=c++1y -O0 -o - %s         | FileCheck --check-prefix=GNU --check-prefix=G32 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple x86_64-windows-gnu                     -emit-llvm -std=c++1y -O0 -o - %s         | FileCheck --check-prefix=GNU --check-prefix=G64 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple i686-windows-msvc -fms-compatibility   -emit-llvm -std=c++1y -O1 -o - %s -DMSABI | FileCheck --check-prefix=MO1 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -disable-llvm-passes -triple i686-windows-gnu                       -emit-llvm -std=c++1y -O1 -o - %s         | FileCheck --check-prefix=GO1 %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Imported {};
struct ExplicitDecl_Imported {};
struct ExplicitInst_Imported {};
struct ExplicitSpec_Imported {};
struct ExplicitSpec_Def_Imported {};
struct ExplicitSpec_InlineDef_Imported {};
struct ExplicitSpec_NotImported {};

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USE(func) void UNIQ(use)() { func(); }
#define USEMV(cls, var) int UNIQ(use)() { return ref(cls::var); }
#define USEMF(cls, fun) template<> void useMemFun<__LINE__, cls>() { cls().fun(); }
#define USESPECIALS(cls) void UNIQ(use)() { useSpecials<cls>(); }

template<typename T>
T ref(T const& v) { return v; }

template<int Line, typename T>
void useMemFun();

template<typename T>
void useSpecials() {
  T v; // Default constructor

  T c1(static_cast<const T&>(v)); // Copy constructor
  T c2 = static_cast<const T&>(v); // Copy constructor
  T c3;
  c3 = static_cast<const T&>(v); // Copy assignment

  T m1(static_cast<T&&>(v)); // Move constructor
  T m2 = static_cast<T&&>(v); // Move constructor
  T m3;
  m3 = static_cast<T&&>(v); // Move assignment
}

// Used to force non-trivial special members.
struct __declspec(dllimport) ForceNonTrivial {
  ForceNonTrivial();
  ~ForceNonTrivial();
  ForceNonTrivial(const ForceNonTrivial&);
  ForceNonTrivial& operator=(const ForceNonTrivial&);
  ForceNonTrivial(ForceNonTrivial&&);
  ForceNonTrivial& operator=(ForceNonTrivial&&);
};



//===----------------------------------------------------------------------===//
// Class members
//===----------------------------------------------------------------------===//

// Import individual members of a class.
struct ImportMembers {
  struct Nested;

  // M32-DAG: define  dso_local dllexport   x86_thiscallcc void @"?normalDef@ImportMembers@@QAEXXZ"(ptr {{[^,]*}} %this)
  // M64-DAG: define  dso_local dllexport                  void @"?normalDef@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}} %this)
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalDecl@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalDecl@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInclass@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInclass@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInlineDef@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInlineDef@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInlineDecl@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInlineDecl@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: define  dso_local             x86_thiscallcc void @_ZN13ImportMembers9normalDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define  dso_local                            void @_ZN13ImportMembers9normalDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: declare           dllimport   x86_thiscallcc void @_ZN13ImportMembers10normalDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare           dllimport                  void @_ZN13ImportMembers10normalDeclEv(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers13normalInclassEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers13normalInclassEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers15normalInlineDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers15normalInlineDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers16normalInlineDeclEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                 void @_ZN13ImportMembers16normalInlineDeclEv(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInclass@ImportMembers@@QAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInlineDef@ImportMembers@@QAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInlineDecl@ImportMembers@@QAEXXZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers13normalInclassEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers15normalInlineDefEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers16normalInlineDeclEv(
  __declspec(dllimport)                void normalDef(); // dllimport ignored
  __declspec(dllimport)                void normalDecl();
  __declspec(dllimport)                void normalInclass() {}
  __declspec(dllimport)                void normalInlineDef();
  __declspec(dllimport)         inline void normalInlineDecl();

  // M32-DAG: define  dso_local dllexport   x86_thiscallcc void @"?virtualDef@ImportMembers@@UAEXXZ"(ptr {{[^,]*}} %this)
  // M64-DAG: define  dso_local dllexport                  void @"?virtualDef@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}} %this)
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualDecl@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualDecl@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInclass@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInclass@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInlineDef@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInlineDef@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInlineDecl@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInlineDecl@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: define  dso_local             x86_thiscallcc void @_ZN13ImportMembers10virtualDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define  dso_local                            void @_ZN13ImportMembers10virtualDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: declare dllimport   x86_thiscallcc void @_ZN13ImportMembers11virtualDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                  void @_ZN13ImportMembers11virtualDeclEv(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers14virtualInclassEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers14virtualInclassEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers16virtualInlineDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers16virtualInlineDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers17virtualInlineDeclEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers17virtualInlineDeclEv(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInclass@ImportMembers@@UAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInlineDef@ImportMembers@@UAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInlineDecl@ImportMembers@@UAEXXZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers14virtualInclassEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers16virtualInlineDefEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers17virtualInlineDeclEv(
  __declspec(dllimport) virtual        void virtualDef(); // dllimport ignored
  __declspec(dllimport) virtual        void virtualDecl();
  __declspec(dllimport) virtual        void virtualInclass() {}
  __declspec(dllimport) virtual        void virtualInlineDef();
  __declspec(dllimport) virtual inline void virtualInlineDecl();

  // MSC-DAG: define  dso_local dllexport                void @"?staticDef@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticDecl@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInclass@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInlineDef@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInlineDecl@ImportMembers@@SAXXZ"()
  // GNU-DAG: define  dso_local                          void @_ZN13ImportMembers9staticDefEv()
  // GNU-DAG: declare           dllimport                void @_ZN13ImportMembers10staticDeclEv()
  // GNU-DAG: define linkonce_odr dso_local               void @_ZN13ImportMembers13staticInclassEv()
  // GNU-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers15staticInlineDefEv()
  // GNU-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers16staticInlineDeclEv()
  // MO1-DAG: define available_externally dllimport void @"?staticInclass@ImportMembers@@SAXXZ"()
  // MO1-DAG: define available_externally dllimport void @"?staticInlineDef@ImportMembers@@SAXXZ"()
  // MO1-DAG: define available_externally dllimport void @"?staticInlineDecl@ImportMembers@@SAXXZ"()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers13staticInclassEv()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers15staticInlineDefEv()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers16staticInlineDeclEv()
  __declspec(dllimport) static         void staticDef(); // dllimport ignored
  __declspec(dllimport) static         void staticDecl();
  __declspec(dllimport) static         void staticInclass() {}
  __declspec(dllimport) static         void staticInlineDef();
  __declspec(dllimport) static  inline void staticInlineDecl();

  // M32-DAG: declare dllimport x86_thiscallcc void @"?protectedNormalDecl@ImportMembers@@IAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                void @"?protectedNormalDecl@ImportMembers@@IEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void @_ZN13ImportMembers19protectedNormalDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                void @_ZN13ImportMembers19protectedNormalDeclEv(ptr {{[^,]*}})
  // MSC-DAG: declare dllimport                void @"?protectedStaticDecl@ImportMembers@@KAXXZ"()
  // GNU-DAG: declare dllimport                void @_ZN13ImportMembers19protectedStaticDeclEv()
protected:
  __declspec(dllimport)                void protectedNormalDecl();
  __declspec(dllimport) static         void protectedStaticDecl();

  // M32-DAG: declare dllimport x86_thiscallcc void @"?privateNormalDecl@ImportMembers@@AAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                void @"?privateNormalDecl@ImportMembers@@AEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void @_ZN13ImportMembers17privateNormalDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                void @_ZN13ImportMembers17privateNormalDeclEv(ptr {{[^,]*}})
  // MSC-DAG: declare dllimport                void @"?privateStaticDecl@ImportMembers@@CAXXZ"()
  // GNU-DAG: declare dllimport                void @_ZN13ImportMembers17privateStaticDeclEv()
private:
  __declspec(dllimport)                void privateNormalDecl();
  __declspec(dllimport) static         void privateStaticDecl();

  // M32-DAG: declare dso_local          x86_thiscallcc void @"?ignored@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dso_local                         void @"?ignored@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dso_local          x86_thiscallcc void @_ZN13ImportMembers7ignoredEv(ptr {{[^,]*}})
  // G64-DAG: declare dso_local                         void @_ZN13ImportMembers7ignoredEv(ptr {{[^,]*}})
public:
  void ignored();

  // MSC-DAG: @"?StaticField@ImportMembers@@2HA"               = external dllimport global i32
  // MSC-DAG: @"?StaticConstField@ImportMembers@@2HB"          = external dllimport constant i32
  // MSC-DAG: @"?StaticConstFieldEqualInit@ImportMembers@@2HB" = available_externally dllimport constant i32 1, align 4
  // MSC-DAG: @"?StaticConstFieldBraceInit@ImportMembers@@2HB" = available_externally dllimport constant i32 1, align 4
  // MSC-DAG: @"?ConstexprField@ImportMembers@@2HB"            = available_externally dllimport constant i32 1, align 4
  // GNU-DAG: @_ZN13ImportMembers11StaticFieldE                   = external dllimport global i32
  // GNU-DAG: @_ZN13ImportMembers16StaticConstFieldE              = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers25StaticConstFieldEqualInitE     = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers25StaticConstFieldBraceInitE     = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers14ConstexprFieldE                = external dllimport constant i32
  __declspec(dllimport) static         int  StaticField;
  __declspec(dllimport) static  const  int  StaticConstField;
  __declspec(dllimport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllimport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllimport) constexpr static int ConstexprField = 1;

  template<int Line, typename T> friend void useMemFun();
};

       void ImportMembers::normalDef() {} // dllimport ignored
inline void ImportMembers::normalInlineDef() {}
       void ImportMembers::normalInlineDecl() {}
       void ImportMembers::virtualDef() {} // dllimport ignored
inline void ImportMembers::virtualInlineDef() {}
       void ImportMembers::virtualInlineDecl() {}
       void ImportMembers::staticDef() {} // dllimport ignored
inline void ImportMembers::staticInlineDef() {}
       void ImportMembers::staticInlineDecl() {}

USEMF(ImportMembers, normalDef)
USEMF(ImportMembers, normalDecl)
USEMF(ImportMembers, normalInclass)
USEMF(ImportMembers, normalInlineDef)
USEMF(ImportMembers, normalInlineDecl)
USEMF(ImportMembers, virtualDef)
USEMF(ImportMembers, virtualDecl)
USEMF(ImportMembers, virtualInclass)
USEMF(ImportMembers, virtualInlineDef)
USEMF(ImportMembers, virtualInlineDecl)
USEMF(ImportMembers, staticDef)
USEMF(ImportMembers, staticDecl)
USEMF(ImportMembers, staticInclass)
USEMF(ImportMembers, staticInlineDef)
USEMF(ImportMembers, staticInlineDecl)
USEMF(ImportMembers, protectedNormalDecl)
USEMF(ImportMembers, protectedStaticDecl)
USEMF(ImportMembers, privateNormalDecl)
USEMF(ImportMembers, privateStaticDecl)
USEMF(ImportMembers, ignored)

USEMV(ImportMembers, StaticField)
USEMV(ImportMembers, StaticConstField)
USEMV(ImportMembers, StaticConstFieldEqualInit)
USEMV(ImportMembers, StaticConstFieldBraceInit)
USEMV(ImportMembers, ConstexprField)


// Import individual members of a nested class.
struct ImportMembers::Nested {
  // M32-DAG: define  dso_local dllexport   x86_thiscallcc void @"?normalDef@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}} %this)
  // M64-DAG: define  dso_local dllexport                  void @"?normalDef@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}} %this)
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalDecl@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalDecl@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInclass@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInclass@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInlineDef@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInlineDef@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?normalInlineDecl@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?normalInlineDecl@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: define  dso_local             x86_thiscallcc void @_ZN13ImportMembers6Nested9normalDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define  dso_local                            void @_ZN13ImportMembers6Nested9normalDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: declare dllimport   x86_thiscallcc void @_ZN13ImportMembers6Nested10normalDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                  void @_ZN13ImportMembers6Nested10normalDeclEv(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested13normalInclassEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested13normalInclassEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested15normalInlineDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested15normalInlineDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested16normalInlineDeclEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested16normalInlineDeclEv(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInclass@Nested@ImportMembers@@QAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInlineDef@Nested@ImportMembers@@QAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?normalInlineDecl@Nested@ImportMembers@@QAEXXZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested13normalInclassEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested15normalInlineDefEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested16normalInlineDeclEv(
  __declspec(dllimport)                void normalDef(); // dllimport ignored
  __declspec(dllimport)                void normalDecl();
  __declspec(dllimport)                void normalInclass() {}
  __declspec(dllimport)                void normalInlineDef();
  __declspec(dllimport)         inline void normalInlineDecl();

  // M32-DAG: define  dso_local dllexport   x86_thiscallcc void @"?virtualDef@Nested@ImportMembers@@UAEXXZ"(ptr {{[^,]*}} %this)
  // M64-DAG: define  dso_local dllexport                  void @"?virtualDef@Nested@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}} %this)
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualDecl@Nested@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualDecl@Nested@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInclass@Nested@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInclass@Nested@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInlineDef@Nested@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInlineDef@Nested@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // M32-DAG: declare           dllimport   x86_thiscallcc void @"?virtualInlineDecl@Nested@ImportMembers@@UAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare           dllimport                  void @"?virtualInlineDecl@Nested@ImportMembers@@UEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: define  dso_local             x86_thiscallcc void @_ZN13ImportMembers6Nested10virtualDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define  dso_local                            void @_ZN13ImportMembers6Nested10virtualDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: declare dllimport   x86_thiscallcc void @_ZN13ImportMembers6Nested11virtualDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                  void @_ZN13ImportMembers6Nested11virtualDeclEv(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested14virtualInclassEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested14virtualInclassEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested16virtualInlineDefEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested16virtualInlineDefEv(ptr {{[^,]*}} %this)
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN13ImportMembers6Nested17virtualInlineDeclEv(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN13ImportMembers6Nested17virtualInlineDeclEv(ptr {{[^,]*}} %this)

  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInclass@Nested@ImportMembers@@UAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInlineDef@Nested@ImportMembers@@UAEXXZ"(
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"?virtualInlineDecl@Nested@ImportMembers@@UAEXXZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc                   void @_ZN13ImportMembers6Nested14virtualInclassEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc                   void @_ZN13ImportMembers6Nested16virtualInlineDefEv(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc                   void @_ZN13ImportMembers6Nested17virtualInlineDeclEv(
  __declspec(dllimport) virtual        void virtualDef(); // dllimport ignored
  __declspec(dllimport) virtual        void virtualDecl();
  __declspec(dllimport) virtual        void virtualInclass() {}
  __declspec(dllimport) virtual        void virtualInlineDef();
  __declspec(dllimport) virtual inline void virtualInlineDecl();

  // MSC-DAG: define  dso_local dllexport                void @"?staticDef@Nested@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticDecl@Nested@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInclass@Nested@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInlineDef@Nested@ImportMembers@@SAXXZ"()
  // MSC-DAG: declare           dllimport                void @"?staticInlineDecl@Nested@ImportMembers@@SAXXZ"()
  // GNU-DAG: define  dso_local                          void @_ZN13ImportMembers6Nested9staticDefEv()
  // GNU-DAG: declare           dllimport                void @_ZN13ImportMembers6Nested10staticDeclEv()
  // GNU-DAG: define linkonce_odr dso_local               void @_ZN13ImportMembers6Nested13staticInclassEv()
  // GNU-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers6Nested15staticInlineDefEv()
  // GNU-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers6Nested16staticInlineDeclEv()
  // MO1-DAG: define available_externally dllimport void @"?staticInclass@Nested@ImportMembers@@SAXXZ"()
  // MO1-DAG: define available_externally dllimport void @"?staticInlineDef@Nested@ImportMembers@@SAXXZ"()
  // MO1-DAG: define available_externally dllimport void @"?staticInlineDecl@Nested@ImportMembers@@SAXXZ"()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers6Nested13staticInclassEv()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers6Nested15staticInlineDefEv()
  // GO1-DAG: define linkonce_odr dso_local              void @_ZN13ImportMembers6Nested16staticInlineDeclEv()
  __declspec(dllimport) static         void staticDef(); // dllimport ignored
  __declspec(dllimport) static         void staticDecl();
  __declspec(dllimport) static         void staticInclass() {}
  __declspec(dllimport) static         void staticInlineDef();
  __declspec(dllimport) static  inline void staticInlineDecl();

  // M32-DAG: declare dllimport x86_thiscallcc void @"?protectedNormalDecl@Nested@ImportMembers@@IAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                void @"?protectedNormalDecl@Nested@ImportMembers@@IEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void @_ZN13ImportMembers6Nested19protectedNormalDeclEv(ptr {{[^,]*}}
  // G64-DAG: declare dllimport                void @_ZN13ImportMembers6Nested19protectedNormalDeclEv(ptr {{[^,]*}})
  // MSC-DAG: declare dllimport                void @"?protectedStaticDecl@Nested@ImportMembers@@KAXXZ"()
  // GNU-DAG: declare dllimport                void @_ZN13ImportMembers6Nested19protectedStaticDeclEv()
protected:
  __declspec(dllimport)                void protectedNormalDecl();
  __declspec(dllimport) static         void protectedStaticDecl();

  // M32-DAG: declare dllimport x86_thiscallcc void @"?privateNormalDecl@Nested@ImportMembers@@AAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                void @"?privateNormalDecl@Nested@ImportMembers@@AEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void @_ZN13ImportMembers6Nested17privateNormalDeclEv(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                void @_ZN13ImportMembers6Nested17privateNormalDeclEv(ptr {{[^,]*}})
  // MSC-DAG: declare dllimport                void @"?privateStaticDecl@Nested@ImportMembers@@CAXXZ"()
  // GNU-DAG: declare dllimport                void @_ZN13ImportMembers6Nested17privateStaticDeclEv()
private:
  __declspec(dllimport)                void privateNormalDecl();
  __declspec(dllimport) static         void privateStaticDecl();

  // M32-DAG: declare dso_local           x86_thiscallcc void @"?ignored@Nested@ImportMembers@@QAEXXZ"(ptr {{[^,]*}})
  // M64-DAG: declare dso_local                          void @"?ignored@Nested@ImportMembers@@QEAAXXZ"(ptr {{[^,]*}})
  // G32-DAG: declare dso_local           x86_thiscallcc void @_ZN13ImportMembers6Nested7ignoredEv(ptr {{[^,]*}})
  // G64-DAG: declare dso_local                          void @_ZN13ImportMembers6Nested7ignoredEv(ptr {{[^,]*}})
public:
  void ignored();

  // MSC-DAG: @"?StaticField@Nested@ImportMembers@@2HA"               = external dllimport global i32
  // MSC-DAG: @"?StaticConstField@Nested@ImportMembers@@2HB"          = external dllimport constant i32
  // MSC-DAG: @"?StaticConstFieldEqualInit@Nested@ImportMembers@@2HB" = available_externally dllimport constant i32 1, align 4
  // MSC-DAG: @"?StaticConstFieldBraceInit@Nested@ImportMembers@@2HB" = available_externally dllimport constant i32 1, align 4
  // MSC-DAG: @"?ConstexprField@Nested@ImportMembers@@2HB"            = available_externally dllimport constant i32 1, align 4
  // GNU-DAG: @_ZN13ImportMembers6Nested11StaticFieldE                   = external dllimport global i32
  // GNU-DAG: @_ZN13ImportMembers6Nested16StaticConstFieldE              = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers6Nested25StaticConstFieldEqualInitE     = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers6Nested25StaticConstFieldBraceInitE     = external dllimport constant i32
  // GNU-DAG: @_ZN13ImportMembers6Nested14ConstexprFieldE                = external dllimport constant i32
  __declspec(dllimport) static         int  StaticField;
  __declspec(dllimport) static  const  int  StaticConstField;
  __declspec(dllimport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllimport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllimport) constexpr static int ConstexprField = 1;

  template<int Line, typename T> friend void useMemFun();
};

       void ImportMembers::Nested::normalDef() {} // dllimport ignored
inline void ImportMembers::Nested::normalInlineDef() {}
       void ImportMembers::Nested::normalInlineDecl() {}
       void ImportMembers::Nested::virtualDef() {} // dllimport ignored
inline void ImportMembers::Nested::virtualInlineDef() {}
       void ImportMembers::Nested::virtualInlineDecl() {}
       void ImportMembers::Nested::staticDef() {} // dllimport ignored
inline void ImportMembers::Nested::staticInlineDef() {}
       void ImportMembers::Nested::staticInlineDecl() {}

USEMF(ImportMembers::Nested, normalDef)
USEMF(ImportMembers::Nested, normalDecl)
USEMF(ImportMembers::Nested, normalInclass)
USEMF(ImportMembers::Nested, normalInlineDef)
USEMF(ImportMembers::Nested, normalInlineDecl)
USEMF(ImportMembers::Nested, virtualDef)
USEMF(ImportMembers::Nested, virtualDecl)
USEMF(ImportMembers::Nested, virtualInclass)
USEMF(ImportMembers::Nested, virtualInlineDef)
USEMF(ImportMembers::Nested, virtualInlineDecl)
USEMF(ImportMembers::Nested, staticDef)
USEMF(ImportMembers::Nested, staticDecl)
USEMF(ImportMembers::Nested, staticInclass)
USEMF(ImportMembers::Nested, staticInlineDef)
USEMF(ImportMembers::Nested, staticInlineDecl)
USEMF(ImportMembers::Nested, protectedNormalDecl)
USEMF(ImportMembers::Nested, protectedStaticDecl)
USEMF(ImportMembers::Nested, privateNormalDecl)
USEMF(ImportMembers::Nested, privateStaticDecl)
USEMF(ImportMembers::Nested, ignored)

USEMV(ImportMembers::Nested, StaticField)
USEMV(ImportMembers::Nested, StaticConstField)
USEMV(ImportMembers::Nested, StaticConstFieldEqualInit)
USEMV(ImportMembers::Nested, StaticConstFieldBraceInit)
USEMV(ImportMembers::Nested, ConstexprField)


// Import special member functions.
struct ImportSpecials {
  // M32-DAG: declare dllimport x86_thiscallcc ptr @"??0ImportSpecials@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // M64-DAG: declare dllimport                ptr @"??0ImportSpecials@@QEAA@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void                    @_ZN14ImportSpecialsC1Ev(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                void                    @_ZN14ImportSpecialsC1Ev(ptr {{[^,]*}})
  __declspec(dllimport) ImportSpecials();

  // M32-DAG: declare dllimport x86_thiscallcc void @"??1ImportSpecials@@QAE@XZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                void @"??1ImportSpecials@@QEAA@XZ"(ptr {{[^,]*}})
  // G32-DAG: declare dllimport x86_thiscallcc void                    @_ZN14ImportSpecialsD1Ev(ptr {{[^,]*}})
  // G64-DAG: declare dllimport                void                    @_ZN14ImportSpecialsD1Ev(ptr {{[^,]*}})
  __declspec(dllimport) ~ImportSpecials();

  // M32-DAG: declare dllimport x86_thiscallcc ptr @"??0ImportSpecials@@QAE@ABU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                ptr @"??0ImportSpecials@@QEAA@AEBU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: declare dllimport x86_thiscallcc void                    @_ZN14ImportSpecialsC1ERKS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G64-DAG: declare dllimport                void                    @_ZN14ImportSpecialsC1ERKS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  __declspec(dllimport) ImportSpecials(const ImportSpecials&);

  // M32-DAG: declare dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportSpecials@@QAEAAU0@ABU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportSpecials@@QEAAAEAU0@AEBU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: declare dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN14ImportSpecialsaSERKS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G64-DAG: declare dllimport                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN14ImportSpecialsaSERKS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  __declspec(dllimport) ImportSpecials& operator=(const ImportSpecials&);

  // M32-DAG: declare dllimport x86_thiscallcc ptr @"??0ImportSpecials@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                ptr @"??0ImportSpecials@@QEAA@$$QEAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: declare dllimport x86_thiscallcc void                    @_ZN14ImportSpecialsC1EOS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G64-DAG: declare dllimport                void                    @_ZN14ImportSpecialsC1EOS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  __declspec(dllimport) ImportSpecials(ImportSpecials&&);

  // M32-DAG: declare dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportSpecials@@QAEAAU0@$$QAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportSpecials@@QEAAAEAU0@$$QEAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: declare dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN14ImportSpecialsaSEOS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G64-DAG: declare dllimport                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN14ImportSpecialsaSEOS_(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  __declspec(dllimport) ImportSpecials& operator=(ImportSpecials&&);
};
USESPECIALS(ImportSpecials)


// Export inline special member functions.
struct ImportInlineSpecials {
  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // M64-DAG: declare dllimport                  ptr @"??0ImportInlineSpecials@@QEAA@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1Ev(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN20ImportInlineSpecialsC1Ev(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@XZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1Ev(
  __declspec(dllimport) ImportInlineSpecials() {}

  // M32-DAG: declare dllimport   x86_thiscallcc void @"??1ImportInlineSpecials@@QAE@XZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                  void @"??1ImportInlineSpecials@@QEAA@XZ"(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsD1Ev(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN20ImportInlineSpecialsD1Ev(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"??1ImportInlineSpecials@@QAE@XZ"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsD1Ev(
  __declspec(dllimport) ~ImportInlineSpecials() {}

  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@ABU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  ptr @"??0ImportInlineSpecials@@QEAA@AEBU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN20ImportInlineSpecialsC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@ABU0@@Z"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1ERKS_(
  __declspec(dllimport) inline ImportInlineSpecials(const ImportInlineSpecials&);

  // M32-DAG: declare dllimport   x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QAEAAU0@ABU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QEAAAEAU0@AEBU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QAEAAU0@ABU0@@Z"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSERKS_(
  __declspec(dllimport) ImportInlineSpecials& operator=(const ImportInlineSpecials&);

  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  ptr @"??0ImportInlineSpecials@@QEAA@$$QEAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN20ImportInlineSpecialsC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportInlineSpecials@@QAE@$$QAU0@@Z"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN20ImportInlineSpecialsC1EOS_(
  __declspec(dllimport) ImportInlineSpecials(ImportInlineSpecials&&) {}

  // M32-DAG: declare dllimport   x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QAEAAU0@$$QAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QEAAAEAU0@$$QEAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportInlineSpecials@@QAEAAU0@$$QAU0@@Z"(
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN20ImportInlineSpecialsaSEOS_(
  __declspec(dllimport) ImportInlineSpecials& operator=(ImportInlineSpecials&&) { return *this; }
};
ImportInlineSpecials::ImportInlineSpecials(const ImportInlineSpecials&) {}
inline ImportInlineSpecials& ImportInlineSpecials::operator=(const ImportInlineSpecials&) { return *this; }
USESPECIALS(ImportInlineSpecials)


// Import defaulted member functions.
struct ImportDefaulted {
  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // M64-DAG: declare dllimport                  ptr @"??0ImportDefaulted@@QEAA@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void                     @_ZN15ImportDefaultedC1Ev(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void                     @_ZN15ImportDefaultedC1Ev(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}} %this)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN15ImportDefaultedC1Ev(ptr {{[^,]*}} %this)
  __declspec(dllimport) ImportDefaulted() = default;

  // M32-DAG: declare dllimport   x86_thiscallcc void @"??1ImportDefaulted@@QAE@XZ"(ptr {{[^,]*}})
  // M64-DAG: declare dllimport                  void @"??1ImportDefaulted@@QEAA@XZ"(ptr {{[^,]*}})
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN15ImportDefaultedD1Ev(ptr {{[^,]*}} %this)
  // G64-DAG: define linkonce_odr dso_local                void @_ZN15ImportDefaultedD1Ev(ptr {{[^,]*}} %this)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc void @"??1ImportDefaulted@@QAE@XZ"(ptr {{[^,]*}} %this)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN15ImportDefaultedD1Ev(ptr {{[^,]*}} %this)
  __declspec(dllimport) ~ImportDefaulted() = default;

  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@ABU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  ptr @"??0ImportDefaulted@@QEAA@AEBU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void                     @_ZN15ImportDefaultedC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                void                     @_ZN15ImportDefaultedC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@ABU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN15ImportDefaultedC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  __declspec(dllimport) ImportDefaulted(const ImportDefaulted&) = default;

  // M32-DAG: declare dllimport   x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QAEAAU0@ABU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QEAAAEAU0@AEBU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QAEAAU0@ABU0@@Z"(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  __declspec(dllimport) ImportDefaulted& operator=(const ImportDefaulted&) = default;

  // M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  ptr @"??0ImportDefaulted@@QEAA@$$QEAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc void                     @_ZN15ImportDefaultedC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                void                     @_ZN15ImportDefaultedC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc ptr @"??0ImportDefaulted@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN15ImportDefaultedC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  __declspec(dllimport) ImportDefaulted(ImportDefaulted&&) = default;

  // M32-DAG: declare dllimport   x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QAEAAU0@$$QAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // M64-DAG: declare dllimport                  nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QEAAAEAU0@$$QEAU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
  // G32-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // G64-DAG: define linkonce_odr dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // MO1-DAG: define available_externally dllimport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaulted@@QAEAAU0@$$QAU0@@Z"(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  // GO1-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN15ImportDefaultedaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
  __declspec(dllimport) ImportDefaulted& operator=(ImportDefaulted&&) = default;

  ForceNonTrivial v; // ensure special members are non-trivial
};
USESPECIALS(ImportDefaulted)


// Import defaulted member function definitions.
struct ImportDefaultedDefs {
  __declspec(dllimport) inline ImportDefaultedDefs();
  __declspec(dllimport) inline ~ImportDefaultedDefs();

  __declspec(dllimport) ImportDefaultedDefs(const ImportDefaultedDefs&);
  __declspec(dllimport) ImportDefaultedDefs& operator=(const ImportDefaultedDefs&);

  __declspec(dllimport) ImportDefaultedDefs(ImportDefaultedDefs&&);
  __declspec(dllimport) ImportDefaultedDefs& operator=(ImportDefaultedDefs&&);
};

#ifdef MSABI
// For MinGW, the function will not be dllimport, and we cannot add the attribute now.
// M32-DAG: declare dllimport x86_thiscallcc ptr @"??0ImportDefaultedDefs@@QAE@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
// M64-DAG: declare dllimport                ptr @"??0ImportDefaultedDefs@@QEAA@XZ"(ptr {{[^,]*}} returned {{[^,]*}})
__declspec(dllimport) ImportDefaultedDefs::ImportDefaultedDefs() = default;
#endif

#ifdef MSABI
// For MinGW, the function will not be dllimport, and we cannot add the attribute now.
// M32-DAG: declare dllimport x86_thiscallcc void @"??1ImportDefaultedDefs@@QAE@XZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??1ImportDefaultedDefs@@QEAA@XZ"(ptr {{[^,]*}})
__declspec(dllimport) ImportDefaultedDefs::~ImportDefaultedDefs() = default;
#endif

// M32-DAG: declare dllimport   x86_thiscallcc ptr @"??0ImportDefaultedDefs@@QAE@ABU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
// M64-DAG: declare dllimport                  ptr @"??0ImportDefaultedDefs@@QEAA@AEBU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
// G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN19ImportDefaultedDefsC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G64-DAG: define linkonce_odr dso_local                 void @_ZN19ImportDefaultedDefsC1ERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
inline ImportDefaultedDefs::ImportDefaultedDefs(const ImportDefaultedDefs&) = default;

// M32-DAG: declare dllimport   x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaultedDefs@@QAEAAU0@ABU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
// M64-DAG: declare dllimport                  nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaultedDefs@@QEAAAEAU0@AEBU0@@Z"(ptr {{[^,]*}}, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}))
// G32-DAG: define linkonce_odr dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN19ImportDefaultedDefsaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G64-DAG: define linkonce_odr dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN19ImportDefaultedDefsaSERKS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
inline ImportDefaultedDefs& ImportDefaultedDefs::operator=(const ImportDefaultedDefs&) = default;

// M32-DAG: define dso_local dllexport x86_thiscallcc ptr @"??0ImportDefaultedDefs@@QAE@$$QAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// M64-DAG: define dso_local dllexport                ptr @"??0ImportDefaultedDefs@@QEAA@$$QEAU0@@Z"(ptr {{[^,]*}} returned {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G32-DAG: define dso_local x86_thiscallcc void @_ZN19ImportDefaultedDefsC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G64-DAG: define dso_local                void @_ZN19ImportDefaultedDefsC1EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G32-DAG: define dso_local x86_thiscallcc void @_ZN19ImportDefaultedDefsC2EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G64-DAG: define dso_local                void @_ZN19ImportDefaultedDefsC2EOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
ImportDefaultedDefs::ImportDefaultedDefs(ImportDefaultedDefs&&) = default; // dllimport ignored

// M32-DAG: define dso_local dllexport x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaultedDefs@@QAEAAU0@$$QAU0@@Z"(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// M64-DAG: define dso_local dllexport                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"??4ImportDefaultedDefs@@QEAAAEAU0@$$QEAU0@@Z"(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G32-DAG: define dso_local x86_thiscallcc nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN19ImportDefaultedDefsaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
// G64-DAG: define dso_local                nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN19ImportDefaultedDefsaSEOS_(ptr {{[^,]*}} %this, ptr nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0)
ImportDefaultedDefs& ImportDefaultedDefs::operator=(ImportDefaultedDefs&&) = default; // dllimport ignored

USESPECIALS(ImportDefaultedDefs)


// Import allocation functions.
struct ImportAlloc {
  __declspec(dllimport) void* operator new(__SIZE_TYPE__);
  __declspec(dllimport) void* operator new[](__SIZE_TYPE__);
  __declspec(dllimport) void operator delete(void*);
  __declspec(dllimport) void operator delete[](void*);
};

// M32-DAG: declare dllimport ptr @"??2ImportAlloc@@SAPAXI@Z"(i32)
// M64-DAG: declare dllimport ptr @"??2ImportAlloc@@SAPEAX_K@Z"(i64)
// G32-DAG: declare dllimport ptr @_ZN11ImportAllocnwEj(i32)
// G64-DAG: declare dllimport ptr @_ZN11ImportAllocnwEy(i64)
void UNIQ(use)() { new ImportAlloc(); }

// M32-DAG: declare dllimport ptr @"??_UImportAlloc@@SAPAXI@Z"(i32)
// M64-DAG: declare dllimport ptr @"??_UImportAlloc@@SAPEAX_K@Z"(i64)
// G32-DAG: declare dllimport ptr @_ZN11ImportAllocnaEj(i32)
// G64-DAG: declare dllimport ptr @_ZN11ImportAllocnaEy(i64)
void UNIQ(use)() { new ImportAlloc[1]; }

// M32-DAG: declare dllimport void @"??3ImportAlloc@@SAXPAX@Z"(ptr)
// M64-DAG: declare dllimport void @"??3ImportAlloc@@SAXPEAX@Z"(ptr)
// G32-DAG: declare dllimport void @_ZN11ImportAllocdlEPv(ptr)
// G64-DAG: declare dllimport void @_ZN11ImportAllocdlEPv(ptr)
void UNIQ(use)(ImportAlloc* ptr) { delete ptr; }

// M32-DAG: declare dllimport void @"??_VImportAlloc@@SAXPAX@Z"(ptr)
// M64-DAG: declare dllimport void @"??_VImportAlloc@@SAXPEAX@Z"(ptr)
// G32-DAG: declare dllimport void @_ZN11ImportAllocdaEPv(ptr)
// G64-DAG: declare dllimport void @_ZN11ImportAllocdaEPv(ptr)
void UNIQ(use)(ImportAlloc* ptr) { delete[] ptr; }


//===----------------------------------------------------------------------===//
// Class member templates
//===----------------------------------------------------------------------===//

struct MemFunTmpl {
  template<typename T>                              void normalDef() {}
  template<typename T> __declspec(dllimport)        void importedNormal() {}
  template<typename T>                       static void staticDef() {}
  template<typename T> __declspec(dllimport) static void importedStatic() {}
};

// Import implicit instantiation of an imported member function template.
// M32-DAG: declare dllimport   x86_thiscallcc void @"??$importedNormal@UImplicitInst_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                  void @"??$importedNormal@UImplicitInst_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN10MemFunTmpl14importedNormalI21ImplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define linkonce_odr dso_local                void @_ZN10MemFunTmpl14importedNormalI21ImplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
USEMF(MemFunTmpl, importedNormal<ImplicitInst_Imported>)

// MSC-DAG: declare dllimport                void @"??$importedStatic@UImplicitInst_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define linkonce_odr dso_local              void @_ZN10MemFunTmpl14importedStaticI21ImplicitInst_ImportedEEvv()
USE(MemFunTmpl::importedStatic<ImplicitInst_Imported>)


// Import explicit instantiation declaration of an imported member function
// template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$importedNormal@UExplicitDecl_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$importedNormal@UExplicitDecl_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: declare dso_local x86_thiscallcc           void @_ZN10MemFunTmpl14importedNormalI21ExplicitDecl_ImportedEEvv(ptr {{[^,]*}})
// G64-DAG: declare dso_local                          void @_ZN10MemFunTmpl14importedNormalI21ExplicitDecl_ImportedEEvv(ptr {{[^,]*}})
extern template void MemFunTmpl::importedNormal<ExplicitDecl_Imported>();
USEMF(MemFunTmpl, importedNormal<ExplicitDecl_Imported>)

// MSC-DAG: declare dllimport                void @"??$importedStatic@UExplicitDecl_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: declare dso_local                void @_ZN10MemFunTmpl14importedStaticI21ExplicitDecl_ImportedEEvv()
extern template void MemFunTmpl::importedStatic<ExplicitDecl_Imported>();
USE(MemFunTmpl::importedStatic<ExplicitDecl_Imported>)


// Import explicit instantiation definition of an imported member function
// template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$importedNormal@UExplicitInst_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$importedNormal@UExplicitInst_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: define weak_odr dso_local x86_thiscallcc   void @_ZN10MemFunTmpl14importedNormalI21ExplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define weak_odr dso_local                  void @_ZN10MemFunTmpl14importedNormalI21ExplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
template void MemFunTmpl::importedNormal<ExplicitInst_Imported>();
USEMF(MemFunTmpl, importedNormal<ExplicitInst_Imported>)

// MSC-DAG: declare dllimport                void @"??$importedStatic@UExplicitInst_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dso_local        void @_ZN10MemFunTmpl14importedStaticI21ExplicitInst_ImportedEEvv()
template void MemFunTmpl::importedStatic<ExplicitInst_Imported>();
USE(MemFunTmpl::importedStatic<ExplicitInst_Imported>)


// Import specialization of an imported member function template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$importedNormal@UExplicitSpec_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$importedNormal@UExplicitSpec_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN10MemFunTmpl14importedNormalI21ExplicitSpec_ImportedEEvv(ptr {{[^,]*}})
// G64-DAG: declare dllimport                void @_ZN10MemFunTmpl14importedNormalI21ExplicitSpec_ImportedEEvv(ptr {{[^,]*}})
template<> __declspec(dllimport) void MemFunTmpl::importedNormal<ExplicitSpec_Imported>();
USEMF(MemFunTmpl, importedNormal<ExplicitSpec_Imported>)

// M32-DAG-FIXME: declare dllimport x86_thiscallcc void @"??$importedNormal@UExplicitSpec_Def_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG-FIXME: declare dllimport                void @"??$importedNormal@UExplicitSpec_Def_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
#ifdef MSABI
//template<> __declspec(dllimport) void MemFunTmpl::importedNormal<ExplicitSpec_Def_Imported>() {}
//USEMF(MemFunTmpl, importedNormal<ExplicitSpec_Def_Imported>)
#endif

// M32-DAG: declare dllimport   x86_thiscallcc void @"??$importedNormal@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                  void @"??$importedNormal@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN10MemFunTmpl14importedNormalI31ExplicitSpec_InlineDef_ImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define linkonce_odr dso_local                void @_ZN10MemFunTmpl14importedNormalI31ExplicitSpec_InlineDef_ImportedEEvv(ptr {{[^,]*}} %this)
template<> __declspec(dllimport) inline void MemFunTmpl::importedNormal<ExplicitSpec_InlineDef_Imported>() {}
USEMF(MemFunTmpl, importedNormal<ExplicitSpec_InlineDef_Imported>)


// MSC-DAG: declare dllimport                void @"??$importedStatic@UExplicitSpec_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: declare dllimport                void @_ZN10MemFunTmpl14importedStaticI21ExplicitSpec_ImportedEEvv()
template<> __declspec(dllimport) void MemFunTmpl::importedStatic<ExplicitSpec_Imported>();
USE(MemFunTmpl::importedStatic<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport                void @"??$importedStatic@UExplicitSpec_Def_Imported@@@MemFunTmpl@@SAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void MemFunTmpl::importedStatic<ExplicitSpec_Def_Imported>() {}
//USE(MemFunTmpl::importedStatic<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport                void @"??$importedStatic@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define linkonce_odr dso_local    void @_ZN10MemFunTmpl14importedStaticI31ExplicitSpec_InlineDef_ImportedEEvv()
template<> __declspec(dllimport) inline void MemFunTmpl::importedStatic<ExplicitSpec_InlineDef_Imported>() {}
USE(MemFunTmpl::importedStatic<ExplicitSpec_InlineDef_Imported>)


// Not importing specialization of an imported member function template without
// explicit dllimport.
// M32-DAG: define dso_local x86_thiscallcc void @"??$importedNormal@UExplicitSpec_NotImported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}} %this)
// M64-DAG: define dso_local                void @"??$importedNormal@UExplicitSpec_NotImported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}} %this)
// G32-DAG: define dso_local x86_thiscallcc void @_ZN10MemFunTmpl14importedNormalI24ExplicitSpec_NotImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define dso_local                void @_ZN10MemFunTmpl14importedNormalI24ExplicitSpec_NotImportedEEvv(ptr {{[^,]*}} %this)
template<> void MemFunTmpl::importedNormal<ExplicitSpec_NotImported>() {}
USEMF(MemFunTmpl, importedNormal<ExplicitSpec_NotImported>)

// MSC-DAG: define dso_local                void @"??$importedStatic@UExplicitSpec_NotImported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define dso_local                void @_ZN10MemFunTmpl14importedStaticI24ExplicitSpec_NotImportedEEvv()
template<> void MemFunTmpl::importedStatic<ExplicitSpec_NotImported>() {}
USE(MemFunTmpl::importedStatic<ExplicitSpec_NotImported>)


// Import explicit instantiation declaration of a non-imported member function
// template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$normalDef@UExplicitDecl_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$normalDef@UExplicitDecl_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: declare dso_local x86_thiscallcc           void @_ZN10MemFunTmpl9normalDefI21ExplicitDecl_ImportedEEvv(ptr {{[^,]*}})
// G64-DAG: declare dso_local                          void @_ZN10MemFunTmpl9normalDefI21ExplicitDecl_ImportedEEvv(ptr {{[^,]*}})
extern template __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitDecl_Imported>();
USEMF(MemFunTmpl, normalDef<ExplicitDecl_Imported>)

// MSC-DAG: declare dllimport                void @"??$staticDef@UExplicitDecl_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: declare dso_local                void @_ZN10MemFunTmpl9staticDefI21ExplicitDecl_ImportedEEvv()
extern template __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitDecl_Imported>();
USE(MemFunTmpl::staticDef<ExplicitDecl_Imported>)


// Import explicit instantiation definition of a non-imported member function
// template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$normalDef@UExplicitInst_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$normalDef@UExplicitInst_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: define weak_odr dso_local x86_thiscallcc   void @_ZN10MemFunTmpl9normalDefI21ExplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define weak_odr dso_local                  void @_ZN10MemFunTmpl9normalDefI21ExplicitInst_ImportedEEvv(ptr {{[^,]*}} %this)
template __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitInst_Imported>();
USEMF(MemFunTmpl, normalDef<ExplicitInst_Imported>)

// MSC-DAG: declare dllimport                void @"??$staticDef@UExplicitInst_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define weak_odr dso_local                  void @_ZN10MemFunTmpl9staticDefI21ExplicitInst_ImportedEEvv()
template __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitInst_Imported>();
USE(MemFunTmpl::staticDef<ExplicitInst_Imported>)


// Import specialization of a non-imported member function template.
// M32-DAG: declare dllimport x86_thiscallcc void @"??$normalDef@UExplicitSpec_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                void @"??$normalDef@UExplicitSpec_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: declare dllimport x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI21ExplicitSpec_ImportedEEvv(ptr {{[^,]*}})
// G64-DAG: declare dllimport                void @_ZN10MemFunTmpl9normalDefI21ExplicitSpec_ImportedEEvv(ptr {{[^,]*}})
template<> __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitSpec_Imported>();
USEMF(MemFunTmpl, normalDef<ExplicitSpec_Imported>)

// M32-DAG-FIXME: declare dllimport x86_thiscallcc void @"??$normalDef@UExplicitSpec_Def_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG-FIXME: declare dllimport                void @"??$normalDef@UExplicitSpec_Def_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
#ifdef MSABI
//template<> __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitSpec_Def_Imported>() {}
//USEMF(MemFunTmpl, normalDef<ExplicitSpec_Def_Imported>)
#endif

// M32-DAG: declare dllimport   x86_thiscallcc void @"??$normalDef@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@QAEXXZ"(ptr {{[^,]*}})
// M64-DAG: declare dllimport                  void @"??$normalDef@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@QEAAXXZ"(ptr {{[^,]*}})
// G32-DAG: define linkonce_odr dso_local x86_thiscallcc void @_ZN10MemFunTmpl9normalDefI31ExplicitSpec_InlineDef_ImportedEEvv(ptr {{[^,]*}} %this)
// G64-DAG: define linkonce_odr dso_local                void @_ZN10MemFunTmpl9normalDefI31ExplicitSpec_InlineDef_ImportedEEvv(ptr {{[^,]*}} %this)
template<> __declspec(dllimport) inline void MemFunTmpl::normalDef<ExplicitSpec_InlineDef_Imported>() {}
USEMF(MemFunTmpl, normalDef<ExplicitSpec_InlineDef_Imported>)


// MSC-DAG: declare dllimport void @"??$staticDef@UExplicitSpec_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: declare dllimport void @_ZN10MemFunTmpl9staticDefI21ExplicitSpec_ImportedEEvv()
template<> __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitSpec_Imported>();
USE(MemFunTmpl::staticDef<ExplicitSpec_Imported>)

// MSC-DAG-FIXME: declare dllimport void @"??$staticDef@UExplicitSpec_Def_Imported@@@MemFunTmpl@@SAXXZ"()
#ifdef MSABI
//template<> __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitSpec_Def_Imported>() {}
//USE(MemFunTmpl::staticDef<ExplicitSpec_Def_Imported>)
#endif

// MSC-DAG: declare dllimport void @"??$staticDef@UExplicitSpec_InlineDef_Imported@@@MemFunTmpl@@SAXXZ"()
// GNU-DAG: define linkonce_odr dso_local void @_ZN10MemFunTmpl9staticDefI31ExplicitSpec_InlineDef_ImportedEEvv()
template<> __declspec(dllimport) inline void MemFunTmpl::staticDef<ExplicitSpec_InlineDef_Imported>() {}
USE(MemFunTmpl::staticDef<ExplicitSpec_InlineDef_Imported>)



struct MemVarTmpl {
  template<typename T>                       static const int StaticVar = 1;
  template<typename T> __declspec(dllimport) static const int ImportedStaticVar = 1;
};

// Import implicit instantiation of an imported member variable template.
// MSC-DAG: @"??$ImportedStaticVar@UImplicitInst_Imported@@@MemVarTmpl@@2HB" = available_externally dllimport constant i32 1, align 4
// GNU-DAG: @_ZN10MemVarTmpl17ImportedStaticVarI21ImplicitInst_ImportedEE       = external dllimport constant i32
USEMV(MemVarTmpl, ImportedStaticVar<ImplicitInst_Imported>)

// Import explicit instantiation declaration of an imported member variable
// template.
// MSC-DAG: @"??$ImportedStaticVar@UExplicitDecl_Imported@@@MemVarTmpl@@2HB" = available_externally dllimport constant i32 1
// GNU-DAG: @_ZN10MemVarTmpl17ImportedStaticVarI21ExplicitDecl_ImportedEE       = external dllimport constant i32
extern template const int MemVarTmpl::ImportedStaticVar<ExplicitDecl_Imported>;
USEMV(MemVarTmpl, ImportedStaticVar<ExplicitDecl_Imported>)

// An explicit instantiation definition of an imported member variable template
// cannot be imported because the template must be defined which is illegal. The
// in-class initializer does not count.

// Import specialization of an imported member variable template.
// MSC-DAG: @"??$ImportedStaticVar@UExplicitSpec_Imported@@@MemVarTmpl@@2HB" = external dllimport constant i32
// GNU-DAG: @_ZN10MemVarTmpl17ImportedStaticVarI21ExplicitSpec_ImportedEE       = external dllimport constant i32
template<> __declspec(dllimport) const int MemVarTmpl::ImportedStaticVar<ExplicitSpec_Imported>;
USEMV(MemVarTmpl, ImportedStaticVar<ExplicitSpec_Imported>)

// Not importing specialization of a member variable template without explicit
// dllimport.
// MSC-DAG: @"??$ImportedStaticVar@UExplicitSpec_NotImported@@@MemVarTmpl@@2HB" = external dso_local constant i32
// GNU-DAG: @_ZN10MemVarTmpl17ImportedStaticVarI24ExplicitSpec_NotImportedEE       = external constant i32
template<> const int MemVarTmpl::ImportedStaticVar<ExplicitSpec_NotImported>;
USEMV(MemVarTmpl, ImportedStaticVar<ExplicitSpec_NotImported>)


// Import explicit instantiation declaration of a non-imported member variable
// template.
// MSC-DAG: @"??$StaticVar@UExplicitDecl_Imported@@@MemVarTmpl@@2HB" = available_externally dllimport constant i32 1
// GNU-DAG: @_ZN10MemVarTmpl9StaticVarI21ExplicitDecl_ImportedEE        = external dllimport constant i32
extern template __declspec(dllimport) const int MemVarTmpl::StaticVar<ExplicitDecl_Imported>;
USEMV(MemVarTmpl, StaticVar<ExplicitDecl_Imported>)

// An explicit instantiation definition of a non-imported member variable template
// cannot be imported because the template must be defined which is illegal. The
// in-class initializer does not count.

// Import specialization of a non-imported member variable template.
// MSC-DAG: @"??$StaticVar@UExplicitSpec_Imported@@@MemVarTmpl@@2HB" = external dllimport constant i32
// GNU-DAG: @_ZN10MemVarTmpl9StaticVarI21ExplicitSpec_ImportedEE        = external dllimport constant i32
template<> __declspec(dllimport) const int MemVarTmpl::StaticVar<ExplicitSpec_Imported>;
USEMV(MemVarTmpl, StaticVar<ExplicitSpec_Imported>)


//===----------------------------------------------------------------------===//
// Class template members
//===----------------------------------------------------------------------===//

template <typename> struct ClassTmplMem {
  void __declspec(dllimport) importedNormal();
  static void __declspec(dllimport) importedStatic();
};
// MSVC imports explicit specialization of imported class template member function; MinGW does not.
// M32-DAG: declare dllimport x86_thiscallcc void @"?importedNormal@?$ClassTmplMem@H@@QAEXXZ"
// G32-DAG: declare dso_local x86_thiscallcc void @_ZN12ClassTmplMemIiE14importedNormalEv
template<> void ClassTmplMem<int>::importedNormal();
USEMF(ClassTmplMem<int>, importedNormal);

// M32-DAG: declare dllimport void @"?importedStatic@?$ClassTmplMem@H@@SAXXZ"
// G32-DAG: declare dso_local void @_ZN12ClassTmplMemIiE14importedStaticEv
template<> void ClassTmplMem<int>::importedStatic();
USEMF(ClassTmplMem<int>, importedStatic);
