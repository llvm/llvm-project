// RUN: %clang_cc1 -std=c++11 -triple i386-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-darwin -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple arm64-ehabi -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple i386-windows -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=MSABI --check-prefix=WIN32
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=MSABI --check-prefix=WIN64

// PR12219
struct A { A(int); virtual ~A(); };
struct B : A { using A::A; ~B(); };
B::~B() {}

B b(123);

struct C { template<typename T> C(T); };
struct D : C { using C::C; };
D d(123);

// ITANIUM-LABEL: define void @_ZN1BD2Ev
// ITANIUM-LABEL: define void @_ZN1BD1Ev
// ITANIUM-LABEL: define void @_ZN1BD0Ev
// WIN32-LABEL: define {{.*}}void @"\01??1B@@UAE@XZ"
// WIN64-LABEL: define {{.*}}void @"\01??1B@@UEAA@XZ"

// ITANIUM-LABEL: define linkonce_odr void @_ZN1BCI11AEi(
// ITANIUM: call void @_ZN1BCI21AEi(

// ITANIUM-LABEL: define linkonce_odr void @_ZN1DCI11CIiEET_(
// ITANIUM: call void @_ZN1DCI21CIiEET_(

// WIN32-LABEL: define internal {{.*}} @"\01??0B@@QAE@H@Z"(
// WIN32: call {{.*}} @"\01??0A@@QAE@H@Z"(
// WIN64-LABEL: define internal {{.*}} @"\01??0B@@QEAA@H@Z"(
// WIN64: call {{.*}} @"\01??0A@@QEAA@H@Z"(

// WIN32-LABEL: define internal {{.*}} @"\01??0D@@QAE@H@Z"(
// WIN32: call {{.*}} @"\01??$?0H@C@@QAE@H@Z"
// WIN64-LABEL: define internal {{.*}} @"\01??0D@@QEAA@H@Z"(
// WIN64: call {{.*}} @"\01??$?0H@C@@QEAA@H@Z"

struct Q { Q(int); Q(const Q&); ~Q(); };
struct Z { Z(); Z(int); ~Z(); int n; };

namespace noninline_nonvirt {
  struct A { A(int, Q&&, void *__attribute__((pass_object_size(0)))); int n; };
  struct B : Z, A { Z z; using A::A; };
  B b(1, 2, &b);
  // ITANIUM-LABEL: define {{.*}} @__cxx_global_var_init
  // ITANIUM: call void @_ZN1QC1Ei({{.*}} %[[TMP:.*]], i32 2)
  // ITANIUM: call void @_ZN17noninline_nonvirt1BCI1NS_1AEEiO1QPvU17pass_object_size0({{.*}} @_ZN17noninline_nonvirt1bE, i32 1, {{.*}} %[[TMP]], i8* {{.*}} @_ZN17noninline_nonvirt1bE{{.*}}, i{{32|64}} 12)
  // ITANIUM: call void @_ZN1QD1Ev({{.*}} %[[TMP]])
  // ITANIUM: call i32 @__cxa_atexit(

  // Complete object ctor for B delegates to base object ctor.
  // ITANIUM-LABEL: define linkonce_odr void @_ZN17noninline_nonvirt1BCI1NS_1AEEiO1QPvU17pass_object_size0(
  // ITANIUM: call void @_ZN17noninline_nonvirt1BCI2NS_1AEEiO1QPvU17pass_object_size0({{.*}}, i32 {{.*}}, %{{.*}}* {{.*}}, i8* {{.*}}, i{{32|64}} {{.*}})

  // In MSABI, we don't have ctor variants. B ctor forwards to A ctor.
  // MSABI-LABEL: define internal {{.*}} @"\01??0B@noninline_nonvirt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
  // MSABI: call {{.*}} @"\01??0Z@@Q{{AE|EAA}}@XZ"(
  // MSABI: call {{.*}} @"\01??0A@noninline_nonvirt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
  // MSABI: call {{.*}} @"\01??0Z@@Q{{AE|EAA}}@XZ"(

  struct C : B { using B::B; };
  C c(1, 2, &c);
  // Complete object ctor for C delegates.
  // ITANIUM-LABEL: define linkonce_odr void @_ZN17noninline_nonvirt1CCI1NS_1AEEiO1QPvU17pass_object_size0(
  // ITANIUM: call void @_ZN17noninline_nonvirt1CCI2NS_1AEEiO1QPvU17pass_object_size0({{.*}}, i32 {{.*}}, %{{.*}}* {{.*}}, i8* {{.*}}, i{{32|64}} {{.*}})

  // MSABI-LABEL: define internal {{.*}} @"\01??0C@noninline_nonvirt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
  // MSABI: call {{.*}} @"\01??0B@noninline_nonvirt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
}

namespace noninline_virt {
  struct A { A(int, Q&&, void *__attribute__((pass_object_size(0)))); int n; };
  struct B : Z, virtual A { Z z; using A::A; };
  B b(1, 2, &b);
  // Complete object ctor forwards to A ctor then constructs Zs.
  // ITANIUM-LABEL: define linkonce_odr void @_ZN14noninline_virt1BCI1NS_1AEEiO1QPvU17pass_object_size0(
  // ITANIUM: call void @_ZN14noninline_virt1AC2EiO1QPvU17pass_object_size0({{.*}} %{{.*}}, i32 %{{.*}}, %{{.*}}* {{.*}}, i8* {{.*}}, i{{32|64}} %{{.*}}
  // ITANIUM: call void @_ZN1ZC2Ev(
  // ITANIUM: store {{.*}} @_ZTVN14noninline_virt1BE
  // ITANIUM: call void @_ZN1ZC1Ev(

  // MSABI-LABEL: define internal {{.*}} @"\01??0B@noninline_virt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}}, i32 %{{.*}})
  // MSABI: %[[COMPLETE:.*]] = icmp ne
  // MSABI: br i1 %[[COMPLETE]],
  // MSABI: call {{.*}} @"\01??0A@noninline_virt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
  // MSABI: br
  // MSABI: call {{.*}} @"\01??0Z@@Q{{AE|EAA}}@XZ"(
  // MSABI: call {{.*}} @"\01??0Z@@Q{{AE|EAA}}@XZ"(

  struct C : B { using B::B; };
  C c(1, 2, &c);
  // Complete object ctor forwards to A ctor, then calls B's base inheriting
  // constructor, which takes no arguments other than the this pointer and VTT.
  // ITANIUM_LABEL: define linkonce_odr void @_ZN14noninline_virt1CCI1NS_1AEEiO1QPvU17pass_object_size0(
  // ITANIUM: call void @_ZN14noninline_virt1AC2EiO1QPvU17pass_object_size0({{.*}} %{{.*}}, i32 %{{.*}}, %{{.*}}* {{.*}}, i8* %{{.*}}, i{{32|64}} %{{.*}})
  // ITANIUM: call void @_ZN14noninline_virt1BCI2NS_1AEEiO1QPvU17pass_object_size0(%{{.*}}* %{{.*}}, i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @_ZTTN14noninline_virt1CE, i64 0, i64 1))
  // ITANIUM: store {{.*}} @_ZTVN14noninline_virt1CE

  // C constructor forwards to B constructor and A constructor. We pass the args
  // to both. FIXME: Can we pass undef here instead, for the base object
  // constructor call?
  // MSABI-LABEL: define internal {{.*}} @"\01??0C@noninline_virt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}}, i32 %{{.*}})
  // MSABI: %[[COMPLETE:.*]] = icmp ne
  // MSABI: br i1 %[[COMPLETE]],
  // MSABI: call {{.*}} @"\01??0A@noninline_virt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}})
  // MSABI: br
  // MSABI: call {{.*}} @"\01??0B@noninline_virt@@Q{{AE|EAA}}@H$$Q{{E?}}AUQ@@P{{E?}}AXW4__pass_object_size0@__clang@@@Z"(%{{.*}}, i32{{.*}}, %{{.*}}, i8*{{.*}}, i{{32|64}}{{.*}}, i32 0)
}

// For MSABI only, check that inalloca arguments result in inlining.
namespace inalloca_nonvirt {
  struct A { A(Q, int, Q, Q&&); int n; };
  struct B : Z, A { Z z; using A::A; };
  B b(1, 2, 3, 4);
  // No inlining implied for Itanium.
  // ITANIUM-LABEL: define linkonce_odr void @_ZN16inalloca_nonvirt1BCI1NS_1AEE1QiS1_OS1_(
  // ITANIUM: call void @_ZN16inalloca_nonvirt1BCI2NS_1AEE1QiS1_OS1_(

  // MSABI-LABEL: define internal void @"\01??__Eb@inalloca_nonvirt@@YAXXZ"(

  // On Win32, the inalloca call can't be forwarded so we force inlining.
  // WIN32: %[[TMP:.*]] = alloca
  // WIN32: call i8* @llvm.stacksave()
  // WIN32: %[[ARGMEM:.*]] = alloca inalloca
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"(%{{.*}}* %[[TMP]], i32 4)
  // WIN32: %[[ARG3:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN32: %[[ARG1:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: %[[ARG2:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store i32 2, i32* %[[ARG2]]
  // WIN32: %[[ARG4:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store {{.*}}* %[[TMP]], {{.*}}** %[[ARG4]]
  // WIN32: call {{.*}} @"\01??0A@inalloca_nonvirt@@QAE@UQ@@H0$$QAU2@@Z"(%{{[^,]*}}, <{{.*}}>* inalloca %[[ARGMEM]])
  // WIN32: call void @llvm.stackrestore(
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??_DQ@@QAEXXZ"(

  // On Win64, the Q arguments would be destroyed in the callee. We don't yet
  // support that in the non-inlined case, so we force inlining.
  // WIN64: %[[TMP:.*]] = alloca
  // WIN64: %[[ARG3:.*]] = alloca
  // WIN64: %[[ARG1:.*]] = alloca
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[TMP]], i32 4)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call {{.*}} @"\01??0A@inalloca_nonvirt@@QEAA@UQ@@H0$$QEAU2@@Z"(%{{.*}}, %{{.*}}* %[[ARG1]], i32 2, %{{.*}}* %[[ARG3]], %{{.*}} %[[TMP]])
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call void @"\01??_DQ@@QEAAXXZ"({{.*}}* %[[TMP]])

  struct C : B { using B::B; };
  C c(1, 2, 3, 4);
  // MSABI-LABEL: define internal void @"\01??__Ec@inalloca_nonvirt@@YAXXZ"(

  // On Win32, the inalloca call can't be forwarded so we force inlining.
  // WIN32: %[[TMP:.*]] = alloca
  // WIN32: call i8* @llvm.stacksave()
  // WIN32: %[[ARGMEM:.*]] = alloca inalloca
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"(%{{.*}}* %[[TMP]], i32 4)
  // WIN32: %[[ARG3:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN32: %[[ARG1:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: %[[ARG2:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store i32 2, i32* %[[ARG2]]
  // WIN32: %[[ARG4:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store {{.*}}* %[[TMP]], {{.*}}** %[[ARG4]]
  // WIN32: call {{.*}} @"\01??0A@inalloca_nonvirt@@QAE@UQ@@H0$$QAU2@@Z"(%{{[^,]*}}, <{{.*}}>* inalloca %[[ARGMEM]])
  // WIN32: call void @llvm.stackrestore(
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??_DQ@@QAEXXZ"(

  // On Win64, the Q arguments would be destroyed in the callee. We don't yet
  // support that in the non-inlined case, so we force inlining.
  // WIN64: %[[TMP:.*]] = alloca
  // WIN64: %[[ARG3:.*]] = alloca
  // WIN64: %[[ARG1:.*]] = alloca
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[TMP]], i32 4)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call {{.*}} @"\01??0A@inalloca_nonvirt@@QEAA@UQ@@H0$$QEAU2@@Z"(%{{.*}}, %{{.*}}* %[[ARG1]], i32 2, %{{.*}}* %[[ARG3]], %{{.*}} %[[TMP]])
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call void @"\01??_DQ@@QEAAXXZ"({{.*}}* %[[TMP]])
}

namespace inalloca_virt {
  struct A { A(Q, int, Q, Q&&); int n; };
  struct B : Z, virtual A { Z z; using A::A; };
  B b(1, 2, 3, 4);

  // MSABI-LABEL: define internal void @"\01??__Eb@inalloca_virt@@YAXXZ"(

  // On Win32, the inalloca call can't be forwarded so we force inlining.
  // WIN32: %[[TMP:.*]] = alloca
  // WIN32: call i8* @llvm.stacksave()
  // WIN32: %[[ARGMEM:.*]] = alloca inalloca
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"(%{{.*}}* %[[TMP]], i32 4)
  // WIN32: %[[ARG3:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN32: %[[ARG1:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // FIXME: It's dumb to round-trip this though memory and generate a branch.
  // WIN32: store i32 1, i32* %[[IS_MOST_DERIVED_ADDR:.*]]
  // WIN32: %[[IS_MOST_DERIVED:.*]] = load i32, i32* %[[IS_MOST_DERIVED_ADDR]]
  // WIN32: %[[IS_MOST_DERIVED_i1:.*]] = icmp ne i32 %[[IS_MOST_DERIVED]], 0
  // WIN32: br i1 %[[IS_MOST_DERIVED_i1]]
  //
  // WIN32: store {{.*}} @"\01??_8B@inalloca_virt@@7B@"
  // WIN32: %[[ARG2:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store i32 2, i32* %[[ARG2]]
  // WIN32: %[[ARG4:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store {{.*}}* %[[TMP]], {{.*}}** %[[ARG4]]
  // WIN32: call {{.*}} @"\01??0A@inalloca_virt@@QAE@UQ@@H0$$QAU2@@Z"(%{{[^,]*}}, <{{.*}}>* inalloca %[[ARGMEM]])
  // WIN32: call void @llvm.stackrestore(
  // WIN32: br
  //
  // Note that if we jumped directly to here we would fail to stackrestore and
  // destroy the parameters, but that's not actually possible.
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??_DQ@@QAEXXZ"(

  // On Win64, the Q arguments would be destroyed in the callee. We don't yet
  // support that in the non-inlined case, so we force inlining.
  // WIN64: %[[TMP:.*]] = alloca
  // WIN64: %[[ARG3:.*]] = alloca
  // WIN64: %[[ARG1:.*]] = alloca
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[TMP]], i32 4)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN64: br i1
  // WIN64: call {{.*}} @"\01??0A@inalloca_virt@@QEAA@UQ@@H0$$QEAU2@@Z"(%{{.*}}, %{{.*}}* %[[ARG1]], i32 2, %{{.*}}* %[[ARG3]], %{{.*}} %[[TMP]])
  // WIN64: br
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call void @"\01??_DQ@@QEAAXXZ"({{.*}}* %[[TMP]])

  struct C : B { using B::B; };
  C c(1, 2, 3, 4);
  // ITANIUM-LABEL: define linkonce_odr void @_ZN13inalloca_virt1CD1Ev(

  // MSABI-LABEL: define internal void @"\01??__Ec@inalloca_virt@@YAXXZ"(

  // On Win32, the inalloca call can't be forwarded so we force inlining.
  // WIN32: %[[TMP:.*]] = alloca
  // WIN32: call i8* @llvm.stacksave()
  // WIN32: %[[ARGMEM:.*]] = alloca inalloca
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"(%{{.*}}* %[[TMP]], i32 4)
  // WIN32: %[[ARG3:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN32: %[[ARG1:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: call {{.*}} @"\01??0Q@@QAE@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN32: store i32 1, i32* %[[IS_MOST_DERIVED_ADDR:.*]]
  // WIN32: %[[IS_MOST_DERIVED:.*]] = load i32, i32* %[[IS_MOST_DERIVED_ADDR]]
  // WIN32: %[[IS_MOST_DERIVED_i1:.*]] = icmp ne i32 %[[IS_MOST_DERIVED]], 0
  // WIN32: br i1 %[[IS_MOST_DERIVED_i1]]
  //
  // WIN32: store {{.*}} @"\01??_8C@inalloca_virt@@7B@"
  // WIN32: %[[ARG2:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store i32 2, i32* %[[ARG2]]
  // WIN32: %[[ARG4:.*]] = getelementptr {{.*}} %[[ARGMEM]]
  // WIN32: store {{.*}}* %[[TMP]], {{.*}}** %[[ARG4]]
  // WIN32: call {{.*}} @"\01??0A@inalloca_virt@@QAE@UQ@@H0$$QAU2@@Z"(%{{[^,]*}}, <{{.*}}>* inalloca %[[ARGMEM]])
  // WIN32: call void @llvm.stackrestore(
  // WIN32: br
  //
  // WIN32: store i32 0, i32* %[[IS_MOST_DERIVED_ADDR:.*]]
  // WIN32: %[[IS_MOST_DERIVED:.*]] = load i32, i32* %[[IS_MOST_DERIVED_ADDR]]
  // WIN32: %[[IS_MOST_DERIVED_i1:.*]] = icmp ne i32 %[[IS_MOST_DERIVED]], 0
  // WIN32: br i1 %[[IS_MOST_DERIVED_i1]]
  //
  // Note: this block is unreachable.
  // WIN32: store {{.*}} @"\01??_8B@inalloca_virt@@7B@"
  // WIN32: br
  //
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??0Z@@QAE@XZ"(
  // WIN32: call {{.*}} @"\01??_DQ@@QAEXXZ"(

  // On Win64, the Q arguments would be destroyed in the callee. We don't yet
  // support that in the non-inlined case, so we force inlining.
  // WIN64: %[[TMP:.*]] = alloca
  // WIN64: %[[ARG3:.*]] = alloca
  // WIN64: %[[ARG1:.*]] = alloca
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[TMP]], i32 4)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG3]], i32 3)
  // WIN64: call {{.*}} @"\01??0Q@@QEAA@H@Z"({{.*}}* %[[ARG1]], i32 1)
  // WIN64: br i1
  // WIN64: store {{.*}} @"\01??_8C@inalloca_virt@@7B@"
  // WIN64: call {{.*}} @"\01??0A@inalloca_virt@@QEAA@UQ@@H0$$QEAU2@@Z"(%{{.*}}, %{{.*}}* %[[ARG1]], i32 2, %{{.*}}* %[[ARG3]], %{{.*}} %[[TMP]])
  // WIN64: br
  // WIN64: br i1
  // (Unreachable block)
  // WIN64: store {{.*}} @"\01??_8B@inalloca_virt@@7B@"
  // WIN64: br
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call {{.*}} @"\01??0Z@@QEAA@XZ"(
  // WIN64: call void @"\01??_DQ@@QEAAXXZ"({{.*}}* %[[TMP]])
}

namespace inline_nonvirt {
  struct A { A(Q, int, Q, Q&&, ...); int n; };
  struct B : Z, A { Z z; using A::A; };
  B b(1, 2, 3, 4, 5, 6);
  // Inlined all the way down to the A ctor.
  // ITANIUM-LABEL: define {{.*}} @__cxx_global_var_init
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 1)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 3)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 4)
  // ITANIUM: %[[Z_BASE:.*]] = bitcast %{{.*}}* %[[THIS:.*]] to
  // ITANIUM: call void @_ZN1ZC2Ev(
  // ITANIUM: %[[B_CAST:.*]] = bitcast {{.*}} %[[THIS]]
  // ITANIUM: %[[A_CAST:.*]] = getelementptr {{.*}} %[[B_CAST]], i{{32|64}} 4
  // ITANIUM: %[[A:.*]] = bitcast {{.*}} %[[A_CAST]]
  // ITANIUM: call void ({{.*}}, ...) @_ZN14inline_nonvirt1AC2E1QiS1_OS1_z(%{{.*}}* %[[A]], {{.*}}, i32 2, {{.*}}, {{.*}}, i32 5, i32 6)
  // ITANIUM: %[[Z_MEMBER:.*]] = getelementptr {{.*}} %[[THIS]], i32 0, i32 2
  // ITANIUM: call void @_ZN1ZC1Ev({{.*}} %[[Z_MEMBER]])
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(

  struct C : B { using B::B; };
  C c(1, 2, 3, 4, 5, 6);
  // Inlined all the way down to the A ctor.
  // ITANIUM-LABEL: define {{.*}} @__cxx_global_var_init
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 1)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 3)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 4)
  // ITANIUM: %[[Z_BASE:.*]] = bitcast %{{.*}}* %[[THIS:.*]] to
  // ITANIUM: call void @_ZN1ZC2Ev(
  // ITANIUM: %[[B_CAST:.*]] = bitcast {{.*}} %[[THIS]]
  // ITANIUM: %[[A_CAST:.*]] = getelementptr {{.*}} %[[B_CAST]], i{{32|64}} 4
  // ITANIUM: %[[A:.*]] = bitcast {{.*}} %[[A_CAST]]
  // ITANIUM: call void ({{.*}}, ...) @_ZN14inline_nonvirt1AC2E1QiS1_OS1_z(%{{.*}}* %[[A]], {{.*}}, i32 2, {{.*}}, {{.*}}, i32 5, i32 6)
  // ITANIUM: %[[Z_MEMBER:.*]] = getelementptr {{.*}} %{{.*}}, i32 0, i32 2
  // ITANIUM: call void @_ZN1ZC1Ev({{.*}} %[[Z_MEMBER]])
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
}

namespace inline_virt {
  struct A { A(Q, int, Q, Q&&, ...); int n; };
  struct B : Z, virtual A { Z z; using A::A; };
  B b(1, 2, 3, 4, 5, 6);
  // Inlined all the way down to the A ctor.
  // ITANIUM-LABEL: define {{.*}} @__cxx_global_var_init
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 1)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 3)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 4)
  // ITANIUM: %[[B_CAST:.*]] = bitcast {{.*}} %[[THIS:.*]]
  // ITANIUM: %[[A_CAST:.*]] = getelementptr {{.*}} %[[B_CAST]], i{{32|64}} {{12|16}}
  // ITANIUM: %[[A:.*]] = bitcast {{.*}} %[[A_CAST]]
  // ITANIUM: call void ({{.*}}, ...) @_ZN11inline_virt1AC2E1QiS1_OS1_z(%{{.*}}* %[[A]], {{.*}}, i32 2, {{.*}}, {{.*}}, i32 5, i32 6)
  // ITANIUM: call void @_ZN1ZC2Ev(
  // ITANIUM: call void @_ZN1ZC1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(

  struct C : B { using B::B; };
  C c(1, 2, 3, 4, 5, 6);
  // Inlined all the way down to the A ctor, except that we can just call the
  // B base inheriting constructor to construct that portion (it doesn't need
  // the forwarded arguments).
  // ITANIUM-LABEL: define {{.*}} @__cxx_global_var_init
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 1)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 3)
  // ITANIUM: call void @_ZN1QC1Ei({{.*}}, i32 4)
  // ITANIUM: %[[B_CAST:.*]] = bitcast {{.*}} %[[THIS:.*]]
  // ITANIUM: %[[A_CAST:.*]] = getelementptr {{.*}} %[[B_CAST]], i{{32|64}} {{12|16}}
  // ITANIUM: %[[A:.*]] = bitcast {{.*}} %[[A_CAST]]
  // ITANIUM: call void ({{.*}}, ...) @_ZN11inline_virt1AC2E1QiS1_OS1_z(%{{.*}}* %[[A]], {{.*}}, i32 2, {{.*}}, {{.*}}, i32 5, i32 6)
  // ITANIUM: call void @_ZN11inline_virt1BCI2NS_1AEE1QiS1_OS1_z({{[^,]*}}, i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @_ZTTN11inline_virt1CE, i64 0, i64 1))
  // ITANIUM: store {{.*}} @_ZTVN11inline_virt1CE
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(
  // ITANIUM: call void @_ZN1QD1Ev(

  // B base object inheriting constructor does not get passed arguments.
  // ITANIUM-LABEL: define linkonce_odr void @_ZN11inline_virt1BCI2NS_1AEE1QiS1_OS1_z(
  // ITANIUM-NOT: call
  // ITANIUM: call void @_ZN1ZC2Ev(
  // ITANIUM-NOT: call
  // VTT -> vtable
  // ITANIUM: store
  // ITANIUM-NOT: call
  // ITANIUM: call void @_ZN1ZC1Ev(
  // ITANIUM-NOT: call
  // ITANIUM: }
}

// ITANIUM-LABEL: define linkonce_odr void @_ZN1BCI21AEi(
// ITANIUM: call void @_ZN1AC2Ei(

// ITANIUM-LABEL: define linkonce_odr void @_ZN1DCI21CIiEET_(
// ITANIUM: call void @_ZN1CC2IiEET_(

// ITANIUM-LABEL: define linkonce_odr void @_ZN17noninline_nonvirt1BCI2NS_1AEEiO1QPvU17pass_object_size0(
// ITANIUM: call void @_ZN1ZC2Ev(
// ITANIUM: call void @_ZN17noninline_nonvirt1AC2EiO1QPvU17pass_object_size0(

// ITANIUM-LABEL: define linkonce_odr void @_ZN17noninline_nonvirt1CCI2NS_1AEEiO1QPvU17pass_object_size0(
// ITANIUM: call void @_ZN17noninline_nonvirt1BCI2NS_1AEEiO1QPvU17pass_object_size0(
