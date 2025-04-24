

// RUN: %clang_cc1 -fbounds-safety -Os %s -triple arm64-apple-iphoneos -emit-llvm -o %t-Os.s -opt-record-file %t-Os.opt.yaml -opt-record-format yaml
// RUN: FileCheck --input-file %t-Os.opt.yaml --check-prefixes OPT-REM %s
// XFAIL: true
// FIXME: rdar://137714109

#include <ptrcheck.h>
static inline int *__bidi_indexable is_counted_by(int * __counted_by(n) ptr, unsigned n) {
    return ptr;
}

void ptr_induction_different_step_sign(int* __indexable A, int N) {
  int *t = is_counted_by(&A[0], N);
  int *t_start = t;
  for(int i = N - 1; i >= 0; i--)   {
     *t_start = 1;
     t_start += 1;
  }
}

void ptr_induction_different_step_sign_2(int* __indexable A, int N) {
  int *t = is_counted_by(&A[0], N);
  int *t_end = t + N - 1;
  for(int i = 0; i < N; i++)   {
     *t_end = 1;
     t_end -= 1;
  }
}

void ptr_induction_different_step_size(int* __indexable A, int N) {
  int *t = is_counted_by(&A[0], 2 * N);
  for(int i = 0 ; i < N; i++)   {
     *t = 1;
     t += 2;
  }
}

void ptr_induction_different_step_size2(int* __indexable A, int N) {
  int *t = is_counted_by(&A[0], 3 * N);
  for(int i = 0 ; i < N; i+=2)   {
     *t = 1;
     t += 3;
  }
}

// OPT-REM: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '3'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-phi-step-size
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 11, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '16'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 12, Column: 12 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '4'
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
// OPT-REM-NEXT:       cmp ugt (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 12, Column: 33 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
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
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''zext'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 12, Column: 26 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "other (LLVM IR 'sub')\nother (LLVM IR 'ashr')"
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-check-ptr-le-upper-bound, bounds-safety-check-ptr-ge-lower-bound
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 15, Column: 6 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound, bounds-safety-missed-optimization-phi-step-size, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          'Cannot remove bound checks because the pointer induction variable and loop counter don''t have the same step size. Consider rewriting the loop counter to have the same step size as the pointer induction variable to help the optimizer remove the access bound checks'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '3'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-phi-step-size
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 20, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '15'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 21, Column: 12 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '4'
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
// OPT-REM-NEXT:       cmp ugt (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 21, Column: 33 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
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
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''zext'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 21, Column: 26 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "other (LLVM IR 'sub')\nother (LLVM IR 'ashr')"
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-check-ptr-le-upper-bound, bounds-safety-check-ptr-ge-lower-bound
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 24, Column: 6 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_sign_2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '6'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound, bounds-safety-missed-optimization-phi-step-size, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          'Cannot remove bound checks because the pointer induction variable and loop counter don''t have the same step size. Consider rewriting the loop counter to have the same step size as the pointer induction variable to help the optimizer remove the access bound checks'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '3'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-phi-step-size
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 29, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '16'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 30, Column: 12 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '4'
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
// OPT-REM-NEXT:       cmp ugt (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 30, Column: 35 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          ''
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''shl'')'
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 30, Column: 33 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
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
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''zext'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 30, Column: 26 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "other (LLVM IR 'sub')\nother (LLVM IR 'ashr')"
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-check-ptr-le-upper-bound, bounds-safety-check-ptr-ge-lower-bound
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 32, Column: 6 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound, bounds-safety-missed-optimization-phi-step-size, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          'Cannot remove bound checks because the pointer induction variable and loop counter don''t have the same step size. Consider rewriting the loop counter to have the same step size as the pointer induction variable to help the optimizer remove the access bound checks'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '9'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-generic
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-le-upper-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '4'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '3'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-missed-optimization-phi-step-size
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            AnnotationSummary
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 37, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Annotated '
// OPT-REM-NEXT:   - count:           '16'
// OPT-REM-NEXT:   - String:          ' instructions with '
// OPT-REM-NEXT:   - type:            bounds-safety-total-summary
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 38, Column: 12 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '4'
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
// OPT-REM-NEXT:       cmp ugt (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-NEXT:       cmp ult (LLVM IR 'icmp')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 38, Column: 35 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '1'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          ''
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''mul'')'
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 38, Column: 33 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
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
// OPT-REM-NEXT:   - String:          'other (LLVM IR ''zext'')'
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 38, Column: 26 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-missed-optimization-nuw
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "other (LLVM IR 'sub')\nother (LLVM IR 'ashr')"
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          Check can not be removed because the arithmetic operation might wrap in the unsigned sense. Optimize the check by adding conditions to check for overflow before doing the operation
// OPT-REM-NEXT: ...

// OPT-REM-NEXT: --- !Analysis
// OPT-REM-NEXT: Pass:            annotation-remarks
// OPT-REM-NEXT: Name:            BoundsSafetyCheck
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 0, Column: 0 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '2'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-generic, bounds-safety-check-ptr-le-upper-bound, bounds-safety-check-ptr-ge-lower-bound
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
// OPT-REM-NEXT: DebugLoc:        { File: '{{.*}}bounds-safety-missed-ptr-induction.c', 
// OPT-REM-NEXT:                    Line: 40, Column: 6 }
// OPT-REM-NEXT: Function:        ptr_induction_different_step_size2
// OPT-REM-NEXT: Args:
// OPT-REM-NEXT:   - String:          'Inserted '
// OPT-REM-NEXT:   - count:           '7'
// OPT-REM-NEXT:   - String:          ' LLVM IR instruction'
// OPT-REM-NEXT:   - String:          s
// OPT-REM-NEXT:   - String:          "\n"
// OPT-REM-NEXT:   - String:          "used for:\n"
// OPT-REM-NEXT:   - String:          bounds-safety-check-ptr-le-upper-bound, bounds-safety-missed-optimization-phi-step-size, bounds-safety-check-ptr-ge-lower-bound
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT: {{^[ 	]+$}}
// OPT-REM-NEXT:       instructions:
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:           |
// OPT-REM-NEXT:       other (LLVM IR 'getelementptr')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       cmp ule (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cmp uge (LLVM IR 'icmp')
// OPT-REM-NEXT:       and (LLVM IR 'and')
// OPT-REM-NEXT:       cond branch (LLVM IR 'br')
// OPT-REM-EMPTY: 
// OPT-REM-NEXT:   - String:          "Missed Optimization Info\n"
// OPT-REM-NEXT:   - String:          'Cannot remove bound checks because the pointer induction variable and loop counter don''t have the same step size. Consider rewriting the loop counter to have the same step size as the pointer induction variable to help the optimizer remove the access bound checks'
// OPT-REM-NEXT: ...

// OPT-REM-NOT: --- !Analysis

