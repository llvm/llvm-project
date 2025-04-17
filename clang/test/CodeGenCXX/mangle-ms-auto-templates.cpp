// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19.20 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-windows-msvc | FileCheck --check-prefix=AFTER %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility-version=19.14 -emit-llvm %s -o - -fms-extensions -fdelayed-template-parsing -triple=x86_64-pc-windows-msvc | FileCheck --check-prefix=BEFORE %s

template <auto a>
class AutoParmTemplate {
public:
  AutoParmTemplate() {}
};

template <auto...>
class AutoParmsTemplate {
public:
  AutoParmsTemplate() {}
};

template <auto a>
auto AutoFunc() {
  return a;
}

int Func();
int Func2();

int i;
int j;

void template_mangling() {
  AutoFunc<1>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MH00@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$00@@YA?A?<auto>@@XZ"
  AutoParmTemplate<0> auto_int;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MH0A@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0A@@@QEAA@XZ"
  AutoParmTemplate<'a'> auto_char;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MD0GB@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0GB@@@QEAA@XZ"
  AutoParmTemplate<9223372036854775807LL> int64_max;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$M_J0HPPPPPPPPPPPPPPP@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0HPPPPPPPPPPPPPPP@@@QEAA@XZ"
  AutoParmTemplate<-9223372036854775807LL - 1LL> int64_min;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$M_J0?IAAAAAAAAAAAAAAA@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0?IAAAAAAAAAAAAAAA@@@QEAA@XZ"
  AutoParmTemplate<(unsigned long long)-1> uint64_neg_1;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$M_K0?0@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$0?0@@QEAA@XZ"

  AutoParmsTemplate<0, false, 'a'> c1;
  // AFTER: call {{.*}} @"??0?$AutoParmsTemplate@$MH0A@$M_N0A@$MD0GB@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmsTemplate@$0A@$0A@$0GB@@@QEAA@XZ"
  AutoParmsTemplate<(unsigned long)1, 9223372036854775807LL> c2;
  // AFTER: call {{.*}} @"??0?$AutoParmsTemplate@$MK00$M_J0HPPPPPPPPPPPPPPP@@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmsTemplate@$00$0HPPPPPPPPPPPPPPP@@@QEAA@XZ"

  AutoFunc<&i>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MPEAH1?i@@3HA@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$1?i@@3HA@@YA?A?<auto>@@XZ"

  AutoParmTemplate<&i> auto_int_ptr;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MPEAH1?i@@3HA@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$1?i@@3HA@@QEAA@XZ"

  AutoParmsTemplate<&i, &j> auto_int_ptrs;
  // AFTER: call {{.*}} @"??0?$AutoParmsTemplate@$MPEAH1?i@@3HA$MPEAH1?j@@3HA@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmsTemplate@$1?i@@3HA$1?j@@3HA@@QEAA@XZ"

  AutoFunc<&Func>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MP6AHXZ1?Func@@YAHXZ@@YA?A_PXZ"
  // BEFORE: call {{.*}} @"??$AutoFunc@$1?Func@@YAHXZ@@YA?A?<auto>@@XZ"

  AutoParmTemplate<&Func> auto_func_ptr;
  // AFTER: call {{.*}} @"??0?$AutoParmTemplate@$MP6AHXZ1?Func@@YAHXZ@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmTemplate@$1?Func@@YAHXZ@@QEAA@XZ"

  AutoParmsTemplate<&Func, &Func2> auto_func_ptrs;
  // AFTER: call {{.*}} @"??0?$AutoParmsTemplate@$MP6AHXZ1?Func@@YAHXZ$MP6AHXZ1?Func2@@YAHXZ@@QEAA@XZ"
  // BEFORE: call {{.*}} @"??0?$AutoParmsTemplate@$1?Func@@YAHXZ$1?Func2@@YAHXZ@@QEAA@XZ"

}
