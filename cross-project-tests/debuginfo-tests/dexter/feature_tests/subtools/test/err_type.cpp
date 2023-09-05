// Purpose:
//      Check that parsing bad commands gives a useful error.
//          - Type error (missing args)
//      Check directives are in check.txt to prevent dexter reading any embedded
//      commands.
//
// Note: Despite using 'lldb' as the debugger, lldb is not actually required
//       as the test should finish before lldb would be invoked.
//
// RUN: not %dexter_base test --builder 'clang' --debugger "lldb" \
// RUN:     --cflags "-O0 -g" -v -- %s \
// RUN:     | FileCheck %s --match-full-lines --strict-whitespace
//
// CHECK:parser error:{{.*}}err_type.cpp(21): expected at least two args
// CHECK:// {{Dex}}ExpectWatchValue()
// CHECK:                      ^

int main(){
    return 0;
}
// DexExpectWatchValue()
