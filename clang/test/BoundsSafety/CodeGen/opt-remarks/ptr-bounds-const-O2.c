

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

// IR: %[[UPPER_BOUND:.+]] = getelementptr inbounds nuw i32, ptr %[[ARRAY:.+]], i64 %{{.+}}, !dbg !{{.+}}
// IR-NEXT: %[[PTR:.+]] = getelementptr i8, ptr %[[ARRAY]], i64 24, !dbg ![[LOC_FOO:[0-9]+]]
// IR-NEXT: %[[ONE_PAST_END:.+]] = getelementptr i8, ptr %[[ARRAY]], i64 28, !dbg ![[LOC_FOO]], !annotation ![[ANNOT_LE_UB:[0-9]+]]
// IR-NEXT: %[[UPPER_CHECK:.+]] = icmp ule ptr %[[ONE_PAST_END]], %[[UPPER_BOUND]], !dbg ![[LOC_FOO]], !annotation ![[ANNOT_LE_UB]]
// IR-NEXT: %[[OVERFLOW_CHECK:.+]] = icmp ule ptr %[[PTR]], %[[ONE_PAST_END]], !dbg ![[LOC_FOO]], !annotation ![[ANNOT_LE_UB]]
// IR-NEXT: %[[UPPER_AND_OVERFLOW_CHECK:.+]] = and i1 %[[OVERFLOW_CHECK]], %[[UPPER_CHECK]], !dbg ![[LOC_FOO]], !annotation ![[ANNOT_LE_UB]]
// IR-NEXT: %[[LOWER_CHECK:.+]] = icmp uge ptr %[[PTR]], %[[ARRAY]], !dbg ![[LOC_FOO]], !annotation ![[ANNOT_GE_LB:[0-9]+]]

// FIXME: opt-remarks on `and` and `br` should be `ANNOT_COMBINED` (rdar://109089053)
// IR-NEXT: %[[COMBINED_CHECKS:.+]] = and i1 %[[LOWER_CHECK]], %[[UPPER_AND_OVERFLOW_CHECK]], !dbg ![[LOC_FOO]], !annotation ![[ANNOT_GE_LB]]
// IR-NEXT: br i1 %[[COMBINED_CHECKS]], label %[[FOO_LABEL_CONT:[a-z0-9]+]], label %[[FOO_LABEL_TRAP:[a-z0-9]+]], !dbg ![[LOC_FOO]], !prof !{{[0-9]+}}, !annotation ![[ANNOT_LE_UB]]
//
// IR: [[FOO_LABEL_TRAP]]:
// IR-NEXT: tail call void @llvm.ubsantrap(i8 25) #3, !dbg ![[LOC_FOO]], !annotation ![[ANNOT_COMBINED:[0-9]+]]
// IR-NEXT: unreachable, !dbg ![[LOC_FOO]], !annotation ![[ANNOT_COMBINED]]
//
// IR: [[FOO_LABEL_CONT]]:
// IR-NEXT: store i32 42, ptr %[[PTR]],{{.+}} !dbg ![[LOC_FOO]]
// IR-NEXT: ret

// IR-LABEL: @main
// ...
// IR: tail call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[LOC_0:[0-9]+]], !annotation ![[ANNOT_LE_UB_AND_CONV_TO_COUNT:[0-9]+]]
// IR-NEXT: unreachable, !dbg ![[LOC_0]], !annotation ![[ANNOT_LE_UB_AND_CONV_TO_COUNT]]


// IR-DAG: ![[ANNOT_LE_UB_AND_CONV_TO_COUNT]] = !{!"bounds-safety-generic"}

// IR-DAG: ![[LOC_FOO:[0-9]+]] = !DILocation(line: 10, column: 14, scope: !5)
// IR-DAG: ![[ANNOT_LE_UB]] = !{!"bounds-safety-check-ptr-le-upper-bound"}
// IR-DAG: ![[ANNOT_GE_LB]] = !{!"bounds-safety-check-ptr-ge-lower-bound"}
// IR-DAG: ![[ANNOT_COMBINED]] = !{!"bounds-safety-check-ptr-le-upper-bound", !"bounds-safety-check-ptr-ge-lower-bound"}

// opt-remarks tests generated using `gen-opt-remarks-check-lines.py`

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-const-O2.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
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
// OPT-REM-NEXT:   - count:           '9'
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
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
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
