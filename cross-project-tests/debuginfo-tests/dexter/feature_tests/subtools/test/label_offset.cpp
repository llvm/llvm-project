// Purpose:
//      Check that we can use label-relative line numbers.
//
// XFAIL: system-darwin
// RUN: %dexter_regression_test_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -v -- %s | FileCheck %s
//
// CHECK: label_offset.cpp: (1.0000)

int main() {  // DexLabel('main')
    int var = 0;
    var = var;
    return 0;
}

/*
DexExpectWatchValue('var', '0', from_line=ref('main')+2, to_line=ref('main')+3)
DexExpectProgramState({
    'frames': [
        {
            'location': { 'lineno': ref('main')+2 },
            'watches': { 'var': '0' }
        }
    ]
})
*/
