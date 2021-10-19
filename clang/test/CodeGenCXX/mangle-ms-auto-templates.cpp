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

void template_mangling() {
  AutoFunc<1>();
  // AFTER: call {{.*}} @"??$AutoFunc@$MH00@@YA?A?<auto>@@XZ"
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
}
