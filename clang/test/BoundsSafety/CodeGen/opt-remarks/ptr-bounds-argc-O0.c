

// RUN: %clang_cc1 -fbounds-safety -O0 %s -emit-llvm -o %t-O0.s -opt-record-file %t-O0.opt.yaml -opt-record-format yaml
// RUN: FileCheck --input-file %t-O0.s --check-prefixes IR %s
// RUN: FileCheck --input-file %t-O0.opt.yaml --check-prefixes OPT-REM %s

#include <ptrcheck.h>

void foo(int* __counted_by(n) array, unsigned n, unsigned idx) {
    array[idx] = 42;
}

int main(int argc, char **argv) {
    unsigned n = argc;
    int * a;
    foo(a, n, argc - 1);
}

// IR-LABEL: @foo
// ...
// IR: icmp ule ptr %{{.*}}, %{{.*}}, !dbg ![[LOC_10_16:[0-9]+]], !annotation ![[ANNOT_LE_UB:[0-9]+]]
// IR-NEXT: br i1 %7, label %[[FOO_LABEL_CONT_0:[a-z0-9]+]], label %[[FOO_LABEL_TRAP_0:[a-z0-9]+]], !dbg ![[LOC_10_16]], !prof ![[PROFILE_METADATA:[0-9]+]], !annotation ![[ANNOT_LE_UB]]
// ...

// IR: [[FOO_LABEL_TRAP_0]]:
// IR: call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg !{{[0-9]+}}, !annotation ![[ANNOT_LE_UB]]
// IR-NEXT: unreachable, !dbg !{{.*}}, !annotation ![[ANNOT_LE_UB]]

// IR: [[FOO_LABEL_CONT_0]]:
// IR:   icmp ule ptr %{{.*}}, %{{.*}}, !dbg ![[LOC_10_16]], !annotation ![[ANNOT_LE_UB]]
// IR-NEXT:   br i1 %8, label  %[[FOO_LABEL_CONT_1:[a-z0-9]+]], label %[[FOO_LABEL_TRAP_1:[a-z0-9]+]], !dbg !11, !prof ![[PROFILE_METADATA:[0-9]+]],  !annotation ![[ANNOT_LE_UB]]

// IR: [[FOO_LABEL_TRAP_1]]:
// IR: call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg !{{.*}}, !annotation ![[ANNOT_LE_UB]]
// IR-NEXT: unreachable, !dbg !{{.*}}, !annotation ![[ANNOT_LE_UB]]

// IR:[[FOO_LABEL_CONT_1]]:
// IR-NEXT: icmp uge ptr %{{.*}}, %{{.*}}, !dbg ![[LOC_10_16]], !annotation  ![[ANNOT_GE_LB:[a-z0-9]+]]
// IR-NEXT: br i1 %{{.*}}, label %[[FOO_LABEL_CONT_2:[a-z0-9]+]], label %[[FOO_LABEL_TRAP_2:[a-z0-9]+]], !dbg !11, !prof ![[PROFILE_METADATA:[0-9]+]],  !annotation ![[ANNOT_GE_LB]]

// IR: [[FOO_LABEL_TRAP_2]]:
// IR-NEXT: call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg !{{.*}}, !annotation ![[ANNOT_GE_LB]]
// IR-NEXT: unreachable, !dbg !{{.*}}, !annotation ![[ANNOT_GE_LB]]


// IR-LABEL: @main
// IR: entry
// ...
// IR-DAG: call void @llvm.memset{{.*}}, !annotation ![[ANNOT_AUTO_INIT:[0-9]+]]
// IR: icmp ule {{.*}} !dbg ![[LOC_16_5:[0-9]+]]
// IR: br i1 %{{.*}}, label %[[MAIN_LABEL_CONT:[a-z0-9.]+]], label %[[MAIN_LABEL_TRAP_RES:[a-z0-9.]+]], !dbg ![[LOC_16_5]]

// IR: [[MAIN_LABEL_CONT]]:
// ...
// IR: icmp ule {{.*}} !dbg ![[LOC_16_5]]
// IR: br i1 %{{.*}}, label %[[MAIN_LABEL_CONT2:[a-z0-9.]+]], label %[[MAIN_LABEL_TRAP_RES]], !dbg ![[LOC_16_5]]

// IR: [[MAIN_LABEL_CONT2]]:
// ...
// IR: %[[WIDTH_CHECK_RES:[a-z0-9_]+]] = icmp ule {{.*}} !dbg ![[LOC_16_5]]
// IR: br  label %[[MAIN_LABEL_TRAP_RES]]

// IR: [[MAIN_LABEL_TRAP_RES]]:
// IR: %[[TRAP_RES:[a-z0-9_]+]] = phi i1 [ false, %[[MAIN_LABEL_CONT]] ], [ false, %entry ], [ %[[WIDTH_CHECK_RES]], %[[MAIN_LABEL_CONT2]] ], !dbg ![[TRAP_LOC_MISSING:[0-9]+]], !annotation ![[ANNOT_CONV_TO_COUNT:[0-9]+]]
// IR: br i1 %[[TRAP_RES]], label {{.*}}, label %[[MAIN_LABEL_TRAP:[a-z0-9.]+]], !dbg ![[LOC_16_5]], !prof ![[PROFILE_METADATA]], !annotation ![[ANNOT_CONV_TO_COUNT]]

// IR: [[MAIN_LABEL_TRAP]]:
// IR:   call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[TRAP_LOC_16_5:[0-9]+]], !annotation ![[ANNOT_CONV_TO_COUNT]]
// IR-NEXT: unreachable, !dbg ![[TRAP_LOC_16_5]], !annotation ![[ANNOT_CONV_TO_COUNT]]

// IR-DAG: ![[ANNOT_CONV_TO_COUNT]] = !{!"bounds-safety-generic"}
// IR-DAG: ![[ANNOT_AUTO_INIT]] = !{!"bounds-safety-zero-init"}

// IR-DAG: ![[LOC_10_16]] = !DILocation(line: 10, column: 16{{.*}})
//
// IR-DAG: ![[LOC_16_5]] = !DILocation(line: 16, column: 5

// IR-DAG: [[ANNOT_LE_UB]] = !{!"bounds-safety-check-ptr-le-upper-bound"}
// IR-DAG: [[ANNOT_GE_LB]] = !{!"bounds-safety-check-ptr-ge-lower-bound"}


// opt-remarks tests generated using `gen-opt-remarks-check-lines.py`

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '13'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 10, Column: 16 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '7'
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
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        foo
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 13, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-zero-init
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 13, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '69'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 13, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '70'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 15, Column: 11 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          ''
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-zero-init
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          'call (LLVM IR ''call'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 16, Column: 9 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '58'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       call (LLVM IR 'call')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'ptrtoint')
// OPT-REM-NEXT:       other (LLVM IR 'ptrtoint')
// OPT-REM-NEXT:       other (LLVM IR 'sub')
// OPT-REM-NEXT:       other (LLVM IR 'sdiv')
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 16, Column: 5 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 16, Column: 12 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          ''
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''zext'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          ''
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''phi'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}ptr-bounds-argc-O0.c', 
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
