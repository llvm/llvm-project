

// RUN: %clang_cc1 -triple x86_64-apple-macos -Wno-bounds-safety-init-list -fbounds-safety -O0 %s -emit-llvm -o %t-O0.s -opt-record-file %t-O0.opt.yaml -opt-record-format yaml
// RUN: FileCheck --input-file %t-O0.s --check-prefixes IR %s
// RUN: FileCheck --input-file %t-O0.opt.yaml --check-prefixes OPT-REM %s

#include <ptrcheck.h>

int main(int argc, char **argv) {
    int len = 0;
    char *__single __counted_by(len) dcp = *argv;
    char *__single tp = dcp;
}

// IR: define{{.*}} i32 @main(i32 noundef %argc, ptr noundef %argv)
// IR: store i32 %argc, ptr %[[ARGC_ALLOCA:[a-z0-9.]+]]
// IR: store ptr %argv, ptr %[[ARGV_ALLOCA:[a-z0-9.]+]]
// IR: %[[ARGV:[a-z0-9.]+]] = load ptr, ptr %[[ARGV_ALLOCA]]{{.*}}
// IR: %[[ARGV_DEREF:[a-z0-9.]+]] = load ptr, ptr %[[ARGV]]{{.*}}
// IR: %[[LEN_DEREF:[a-z0-9.]+]] = load i32, ptr %{{.*}}, !dbg ![[LOC_11_20:[0-9]+]]
// IR: %[[LEN_EXT:[a-z0-9.]+]] = sext i32 %[[LEN_DEREF]] to i64, !dbg ![[LOC_11_20]]
// IR: icmp ule ptr %[[ARGV_DEREF]], {{.*}}, !dbg ![[LOC_11_37:[0-9]+]]
// IR: br i1 {{.*}}, label %[[LABEL_CONT:[a-z0-9.]+]], label %[[LABEL_TRAP_RES:[a-z0-9.]+]], !dbg ![[LOC_11_37]]
// ...
// IR: [[LABEL_CONT]]:
// ...
// IR: icmp ule ptr {{.*}}, %[[ARGV_DEREF]], !dbg ![[LOC_11_44:[0-9]+]]
// IR: br i1 %{{.*}}, label %[[LABEL_CONT2:.+]], label %[[LABEL_TRAP_RES]], !dbg ![[LOC_11_44]]

// IR: [[LABEL_CONT2]]:
// ...
// IR: icmp sle i64 %[[LEN_EXT]], {{.*}}, !dbg ![[LOC_11_44]]
// IR: br i1 %{{.*}}, label %[[LABEL_CONT3:.+]], label %[[LABEL_TRAP_RES2:.+]], !dbg ![[LOC_11_44]]

// IR: [[LABEL_CONT3]]:
// IR: %[[LEN_CHECK_RES:[a-z0-9_]+]] = icmp sle i64 0, %[[LEN_EXT]], !dbg ![[LOC_11_44]]
// IR: br label %[[LABEL_TRAP_RES2]]

// IR: [[LABEL_TRAP_RES2]]:
// IR: %[[TRAP_RES2:[a-z0-9_]+]] = phi i1 [ false, %[[LABEL_CONT2]] ], [ %[[LEN_CHECK_RES]], %[[LABEL_CONT3]] ], !dbg ![[TRAP_LOC_MISSING:[0-9]+]]
// IR: br label %[[LABEL_TRAP_RES]]

// IR: [[LABEL_TRAP_RES]]:
// IR: %[[TRAP_RES:[a-z0-9_]+]] = phi i1 [ false, %[[LABEL_CONT]] ], [ false, %entry ], [ %[[TRAP_RES2]], %[[LABEL_TRAP_RES2]] ], !dbg ![[TRAP_LOC_MISSING:[0-9]+]], !annotation ![[ANNOT_CONV_TO_COUNT:[0-9]+]]
// IR: br i1 %[[TRAP_RES]], label %[[LABEL_CONT4:[a-z0-9.]+]], label %[[LABEL_TRAP:[a-z0-9.]+]], !dbg ![[LOC_11_44]], !annotation ![[ANNOT_CONV_TO_COUNT]]

// IR: [[LABEL_TRAP]]:
// IR: call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[TRAP_LOC_11_44:[0-9]+]], !annotation ![[ANNOT_CONV_TO_COUNT]]
// IR-NEXT: unreachable, !dbg ![[TRAP_LOC_11_44]], !annotation ![[ANNOT_CONV_TO_COUNT]]

// IR: [[LABEL_CONT4]]:
// IR: %[[NULL_CHECK_RES:[a-z0-9_]+]] = icmp ne ptr %[[WIDE_PTR:[a-z0-9_.]+]], null, !dbg ![[LOC_TMP2:[0-9]+]], !annotation ![[ANNOT1:[0-9]+]]
// IR: br i1 %[[NULL_CHECK_RES]], label %[[LABEL_CONT5:[a-z0-9.]+]], label %[[LABEL_END:[a-z0-9.]+]], !dbg ![[LOC_TMP:[0-9]+]], !annotation ![[ANNOT2:[0-9]+]]

// IR: [[LABEL_CONT5]]:
// IR: %[[LT_CHECK_RES:[a-z0-9_]+]] = icmp ult ptr %[[WIDE_PTR]], {{.*}}, !dbg ![[LOC_TMP]], !annotation ![[ANNOT3:[0-9]+]]
// IR: br i1 %[[LT_CHECK_RES]], label %[[LABEL_CONT6:[a-z0-9.]+]], label %[[LABEL_TRAP2:[a-z0-9.]+]], !dbg ![[LOC_TMP]], !annotation ![[ANNOT3]]

// IR: [[LABEL_TRAP2]]:
// IR: call void @llvm.ubsantrap(i8 25) #{{[0-9]+}}, !dbg ![[TRAP_LOC2:[0-9]+]], !annotation ![[ANNOT_TRAP:[0-9]+]]
// IR-NEXT: unreachable, !dbg ![[TRAP_LOC2]], !annotation ![[ANNOT_TRAP]]

// IR: [[LABEL_CONT6]]:
// IR: %[[GE_CHECK_RES:[a-z0-9_]+]] = icmp uge ptr %[[WIDE_PTR]], {{.*}}, !dbg ![[LOC_TMP]], !annotation ![[ANNOT4:[0-9]+]]
// IR: br i1 %[[GE_CHECK_RES]], label %[[LABEL_CONT7:[a-z0-9.]+]], label %[[LABEL_TRAP3:[a-z0-9.]+]], !dbg ![[LOC_TMP]], !annotation ![[ANNOT4]]

// IR: [[LABEL_END]]:

// IR-DAG: ![[LOC_11_44]] = !DILocation(line: 11, column: 44
// IR-DAG: ![[TRAP_LOC_11_44]] = !DILocation(line: 0, scope: ![[TRAP_INFO_11_44:[0-9]+]], inlinedAt: ![[LOC_11_44]])
// IR-DAG: ![[TRAP_INFO_11_44]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$"
//
// IR-DAG: ![[LOC_11_20]] = !DILocation(line: 11, column: 20

// IR-DAG: ![[TRAP_LOC_MISSING]] = !DILocation(line: 0, scope: ![[MAIN_SCOPE:[0-9]+]])
// IR-DAG: ![[MAIN_SCOPE]] = distinct !DISubprogram(name: "main", {{.*}} line: 9, {{.*}} scopeLine: 9

// IR-DAG: ![[ANNOT_CONV_TO_COUNT]] = !{!"bounds-safety-generic"}
// IR-DAG: ![[ANNOT_TRAP]] = !{!"bounds-safety-check-ptr-lt-upper-bound"}

// opt-remarks tests generated using `gen-opt-remarks-check-lines.py`

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '43'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-neq-null
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-lt-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 9, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '53'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 44 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '38'
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
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'store')
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       other (LLVM IR 'load')
// OPT-REM-NEXT:       other (LLVM IR 'ptrtoint')
// OPT-REM-NEXT:       other (LLVM IR 'ptrtoint')
// OPT-REM-NEXT:       other (LLVM IR 'sub')
// OPT-REM-NEXT:       cmp sle (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp sle (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
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
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''phi'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
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

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 12, Column: 25 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-neq-null, bounds-safety-check-ptr-lt-upper-bound, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       cmp ne (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        main
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-lt-upper-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-ptr-conversion-O0.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        main
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
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "trap (LLVM IR 'call')\nother (LLVM IR 'unreachable')"
// OPT-REM-NEXT: ...

// OPT-REM-NOT: --- !Analysis
