// RUN: %clang -cc1depscan -o - -cc1-args -cc1 -triple \
// RUN:        x86_64-apple-darwin10 -debug-info-kind=standalone -dwarf-version=4 \
// RUN:        -debugger-tuning=lldb -emit-obj -fcas-backend -fcas-path %t/cas \
// RUN:        -fcas-emit-casid-file -mllvm -cas-friendly-debug-info %s | FileCheck %s --check-prefix=MCCAS_ON

// MCCAS_ON: -mllvm
// MCCAS_ON: -cas-friendly-debug-info
// MCCAS_ON: -fcas-backend
// MCCAS_ON: -fcas-emit-casid-file

// RUN: %clang -cc1depscan -o - -cc1-args -cc1 -triple \
// RUN:        x86_64-apple-darwin10 -debug-info-kind=standalone -dwarf-version=4 \
// RUN:        -debugger-tuning=lldb -emit-llvm -fcas-backend -fcas-path %t/cas \
// RUN:        -fcas-emit-casid-file -mllvm -cas-friendly-debug-info %s | FileCheck %s --check-prefix=MCCAS_OFF 

// MCCAS_OFF-NOT: -mllvm
// MCCAS_OFF-NOT: -cas-friendly-debug-info
// MCCAS_OFF-NOT: -fcas-backend
// MCCAS_OFF-NOT: -fcas-emit-casid-file


int foo(int x) {
    return x+1;
}
