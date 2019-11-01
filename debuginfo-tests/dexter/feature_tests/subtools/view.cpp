// Purpose:
//      Check the `view` subtool works with typical inputs.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:    --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" \
// RUN:    --results %t -- %s
//
// RUN: %dexter_base view %t/view.cpp.dextIR | FileCheck %s
// CHECK: ## BEGIN
// CHECK: ## END
//
// # [TODO] This doesn't run if FileCheck fails!
// RUN: rm -rf %t

int main() {
    int a = 0;
    return 0; //DexLabel('ret')
}
// DexExpectWatchValue('a', '0', on_line='ret')
