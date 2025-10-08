// Tests that we assign appropriate identifiers to indirect calls and targets
// specifically for virtual methods.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=FT %s < %t
// RUN: FileCheck --check-prefix=CST %s < %t

////////////////////////////////////////////////////////////////////////////////
// Class definitions (check for indirect target metadata)

class Base {
  public:
    // FT-DAG: define {{.*}} @_ZN4Base2vfEPc({{.*}} !type [[F_TVF:![0-9]+]]
    virtual int vf(char *a) { return 0; };
  };
  
  class Derived : public Base {
  public:
    // FT-DAG: define {{.*}} @_ZN7Derived2vfEPc({{.*}} !type [[F_TVF]]
    int vf(char *a) override { return 1; };
  };
  
  // FT-DAG: [[F_TVF]] = !{i64 0, !"_ZTSFiPvE.generalized"}
  
  ////////////////////////////////////////////////////////////////////////////////
  // Callsites (check for indirect callsite operand bundles)
  
  // CST-LABEL: define {{.*}} @_Z3foov
  void foo() {
    auto B = Base();
    auto D = Derived();
  
    Base *Bptr = &B;
    Base *BptrToD = &D;
    Derived *Dptr = &D;
  
    auto FpBaseVf = &Base::vf;
    auto FpDerivedVf = &Derived::vf;
  
    // CST: call noundef i32 %{{.*}}, !callee_type [[F_TVF_CT:![0-9]+]]
    (Bptr->*FpBaseVf)(0);
  
    // CST: call noundef i32 %{{.*}}, !callee_type [[F_TVF_CT:![0-9]+]]
    (BptrToD->*FpBaseVf)(0);
  
    // CST: call noundef i32 %{{.*}}, !callee_type [[F_TVF_CT:![0-9]+]]
    (Dptr->*FpBaseVf)(0);
  
    // CST: call noundef i32 %{{.*}}, !callee_type [[F_TVF_CT:![0-9]+]]
    (Dptr->*FpDerivedVf)(0);
  }

  // CST-DAG: [[F_TVF_CT]] = !{[[F_TVF:![0-9]+]]}
  // CST-DAG: [[F_TVF]] = !{i64 0, !"_ZTSFiPvE.generalized"}
