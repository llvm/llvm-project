// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=NOUNDEF --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=NOUNDEF --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -no-enable-noundef-analysis -fclangir -emit-llvm %s -o %t-cir-nonoundef.ll
// RUN: FileCheck --check-prefix=NONOUNDEF --input-file=%t-cir-nonoundef.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -no-enable-noundef-analysis -emit-llvm %s -o %t-nonoundef.ll
// RUN: FileCheck --check-prefix=NONOUNDEF --input-file=%t-nonoundef.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -no-enable-noundef-analysis -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t-nonoundef.cir
// RUN: cir-opt %t-nonoundef.cir -cir-call-conv-lowering=target=x86_64 -o %t-nonoundef-ccl.cir
// RUN: FileCheck --check-prefix=NONOUNDEF-CIR --input-file=%t-nonoundef-ccl.cir %s

// -no-enable-noundef-analysis removes noundef from the arguments that
// calling convention lowering marks noundef: extended integers and indirect
// arguments with and without byval.  The NONOUNDEF-CIR run emits CIR without
// calling convention lowering and runs the pass as a separate command, which
// reads the setting from the module's #cir.lowering_options.

// NONOUNDEF-CIR: cir.lowering_options = #cir.lowering_options<
// NONOUNDEF-CIR-SAME: enable_noundef_attrs = false>

struct Big {
  long a, b, c;
};

struct NonTrivial {
  NonTrivial(const NonTrivial &);
  int x;
};

extern "C" {

void take_char(signed char c) {}

// NOUNDEF:   define dso_local void @take_char(i8 noundef signext %{{.*}})
// NONOUNDEF: define dso_local void @take_char(i8 signext %{{.*}})
// NONOUNDEF-CIR: cir.func {{.*}} @take_char(%{{.*}}: !s8i {llvm.signext})

void take_big(Big b) {}

// NOUNDEF:   define dso_local void @take_big(ptr noundef byval(%struct.Big) align 8 %{{.*}})
// NONOUNDEF: define dso_local void @take_big(ptr byval(%struct.Big) align 8 %{{.*}})
// NONOUNDEF-CIR: cir.func {{.*}} @take_big(%{{.*}}: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big})

void take_non_trivial(NonTrivial n) {}

// NOUNDEF:   define dso_local void @take_non_trivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{.*}})
// NONOUNDEF: define dso_local void @take_non_trivial(ptr nofreeobj align 4 dead_on_return dereferenceable(4) %{{.*}})
// NONOUNDEF-CIR: cir.func {{.*}} @take_non_trivial(%{{.*}}: !cir.ptr<!rec_NonTrivial> {llvm.align = 4 : i64, llvm.dead_on_return = -1 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj})

void ext_char(signed char c);
void ext_big(Big b);
void ext_non_trivial(NonTrivial n);

void callers(signed char c, Big b, NonTrivial n) {
  ext_char(c);
  ext_big(b);
  ext_non_trivial(n);
}

// NOUNDEF-LABEL:   define dso_local void @callers(
// NOUNDEF:           call void @ext_char(i8 noundef signext %{{.*}})
// NOUNDEF:           call void @ext_big(ptr noundef byval(%struct.Big) align 8 %{{.*}})
// NOUNDEF:           call void @ext_non_trivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{.*}})

// NOUNDEF-DAG: declare void @ext_char(i8 noundef signext)
// NOUNDEF-DAG: declare void @ext_big(ptr noundef byval(%struct.Big) align 8)
// NOUNDEF-DAG: declare void @ext_non_trivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4))

// NONOUNDEF-LABEL: define dso_local void @callers(
// NONOUNDEF:         call void @ext_char(i8 signext %{{.*}})
// NONOUNDEF:         call void @ext_big(ptr byval(%struct.Big) align 8 %{{.*}})
// NONOUNDEF:         call void @ext_non_trivial(ptr nofreeobj align 4 dead_on_return dereferenceable(4) %{{.*}})

// NONOUNDEF-DAG: declare void @ext_char(i8 signext)
// NONOUNDEF-DAG: declare void @ext_big(ptr byval(%struct.Big) align 8)
// NONOUNDEF-DAG: declare void @ext_non_trivial(ptr nofreeobj align 4 dead_on_return dereferenceable(4))

// NONOUNDEF-CIR-LABEL: cir.func {{.*}} @callers(
// NONOUNDEF-CIR:         cir.call @ext_char(%{{.*}}) : (!s8i {llvm.signext}) -> ()
// NONOUNDEF-CIR:         cir.call @ext_big(%{{.*}}) : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big}) -> ()
// NONOUNDEF-CIR:         cir.call @ext_non_trivial(%{{.*}}) : (!cir.ptr<!rec_NonTrivial> {llvm.align = 4 : i64, llvm.dead_on_return = -1 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj}) -> ()

// NONOUNDEF-CIR-DAG: cir.func private @ext_char(!s8i {llvm.signext})
// NONOUNDEF-CIR-DAG: cir.func private @ext_big(!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big})
// NONOUNDEF-CIR-DAG: cir.func private @ext_non_trivial(!cir.ptr<!rec_NonTrivial> {llvm.align = 4 : i64, llvm.dead_on_return = -1 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj})

}
