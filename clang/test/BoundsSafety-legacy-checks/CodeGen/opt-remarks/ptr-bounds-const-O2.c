

// RUN: %clang_cc1 -fbounds-safety -O2 %s -emit-llvm -o %t-O2.s -opt-record-file %t-O2.opt.yaml -opt-record-format yaml
// RUN: FileCheck --input-file %t-O2.s --check-prefixes IR %s
// RUN: FileCheck --input-file %t-O2.opt.yaml --check-prefixes OPT-REM %s

#include <ptrcheck.h>

void foo(int* __counted_by(n) array, unsigned n) {
    array[6] = 42;
}

int main() {
    unsigned n = 5;
    int * a;
    foo(a, n);
}

// IR-LABEL: @foo
// ...
// IR-DAG: icmp ult ptr {{.*}}, !dbg {{.+}}, !annotation ![[ANNOT_LT_UB:[0-9]+]]
// IR-DAG: icmp uge ptr {{%.*}}, !dbg {{.+}}, !annotation ![[ANNOT_GE_LB:[0-9]+]]
// IR: [[OR_COND:%.*]] = and i1 %{{[a-z0-9]+}}, %{{[a-z0-9]+}}, !dbg {{.+}}, !annotation ![[ANNOT_GE_LB]]
// IR: br i1 [[OR_COND]], label %{{[a-z0-9]+}}, label %[[FOO_LABEL_TRAP:[a-z0-9]+]], !dbg {{.+}}, !annotation ![[ANNOT_LT_UB]]
// ...
// IR: [[FOO_LABEL_TRAP]]:
// IR:   call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[LOC_10_14:[0-9]+]], !annotation ![[ANNOT_LT_UB_AND_GE_LB:[0-9]+]]
// IR-NEXT: unreachable, !dbg ![[LOC_10_14]], !annotation ![[ANNOT_LT_UB_AND_GE_LB]]


// IR-LABEL: @main
// ...
// IR: tail call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[LOC_0:[0-9]+]], !annotation ![[ANNOT_LE_UB_AND_CONV_TO_COUNT:[0-9]+]]
// IR-NEXT: unreachable, !dbg ![[LOC_0]], !annotation ![[ANNOT_LE_UB_AND_CONV_TO_COUNT]]


// IR-DAG: ![[ANNOT_LT_UB]] = !{!"bounds-safety-check-ptr-lt-upper-bound"}
// IR-DAG: ![[ANNOT_GE_LB]] = !{!"bounds-safety-check-ptr-ge-lower-bound"}
// IR-DAG: ![[ANNOT_LT_UB_AND_GE_LB]] = !{!"bounds-safety-check-ptr-lt-upper-bound", !"bounds-safety-check-ptr-ge-lower-bound"}
// IR-DAG: ![[ANNOT_LE_UB_AND_CONV_TO_COUNT]] = !{!"bounds-safety-generic"}

// opt-remarks tests generated using `gen-opt-remarks-check-lines.py`

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-lt-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 10, Column: 14 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-lt-upper-bound, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       trap (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'unreachable')
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 13, Column: 0 }
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 13, Column: 0 }
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
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
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NOT: --- !Analysis
