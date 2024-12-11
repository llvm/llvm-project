

// RUN: %clang_cc1 -fbounds-safety -O2 %s -emit-llvm -o %t-O2.s -opt-record-file %t-O2.opt.yaml -opt-record-format yaml
// RUN: FileCheck --input-file %t-O2.s --check-prefixes IR %s
// RUN: FileCheck --input-file %t-O2.opt.yaml --check-prefixes OPT-REM %s

#include <ptrcheck.h>

int main(int argc, char **argv) {
    int count_const = 0;
    int * __counted_by(count_const) buff_const = 0;
    count_const = 5;
    buff_const = buff_const;

    int count_rt = argc;
    int * __counted_by(count_rt) buff_rt = 0;
    count_rt = argc;
    buff_rt = buff_rt;
}

// IR: define{{.*}} i32 @main
// IR-NEXT: entry:
// IR-NEXT:   call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[LOC_0:[0-9]+]], !annotation ![[TRAP_REASON:[0-9]+]]
// IR-NEXT:   unreachable, !dbg ![[LOC_0]], !annotation ![[TRAP_REASON]]


// IR-DAG: ![[TRAP_REASON]] = !{!"bounds-safety-generic"}

// opt-remarks tests generated using `gen-opt-remarks-check-lines.py`

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-count-assignment-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-count-assignment-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-count-assignment-O2.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NOT: --- !Analysis
