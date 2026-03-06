// RUN: %clang_cc1 -triple aarch64-pc-windows -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck %s

void may_throw(void);
void puts(const char *);

int main(void) {
	@try {
		may_throw();
        // CHECK: invoke void @may_throw()
        // CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
	}

    // Check that the dispatch block has been emitted correctly. We capture the
    // normal and unwind edge for later checks.
    // CHECK: [[CATCH_DISPATCH]]:
    // CHECK-NEXT: %[[CATCHSWITCH_OUTER:.*]] = catchswitch within none [label %[[CATCH_A:.*]], label %[[CATCH_B:.*]]] unwind label %[[EH_CLEANUP_OUTER_FINALLY:.*]]

    // Check if normal edge leads to a call to the outer finally funclet
    // CHECK: [[INVOKE_CONT]]:
    // CHECK: call void @"?fin$0@0@main@@"

	@catch(id a) {
        // CHECK: %[[CATCHPAD_A:.*]] = catchpad within %[[CATCHSWITCH_OUTER]]
		puts("catch");
        @try {
			may_throw();
            // CHECK: invoke void @may_throw() [ "funclet"(token %{{.*}}) ]
            // CHECK-NEXT: to label %[[INVOKE_CONT_INNER:.*]] unwind label %[[CATCH_DISPATCH_INNER:.*]]

            // CHECK: [[CATCH_DISPATCH_INNER]]:
            // CHECK-NEXT: %{{.*}} = catchswitch within %[[CATCHPAD_A]] [label %[[CATCHPAD_A_INNER:.*]]] unwind label %[[EH_CLEANUP_INNER_FINALLY:.*]]

            // Check if normal edge leads to a call to the inner finally funclet and calls
            // CHECK: [[INVOKE_CONT_INNER]]:
            // CHECK: invoke void @"?fin$1@0@main@@"
		} @catch(...) {
            // CHECK: [[CATCHPAD_A_INNER]]:
            // CHECK: to label %invoke.cont{{[0-9]+}} unwind label %[[EH_CLEANUP_INNER_FINALLY]]
			puts("inner catch all");
		} @finally {
            // CHECK: [[EH_CLEANUP_INNER_FINALLY]]:
            // CHECK: invoke void @"?fin$1@0@main@@"{{.*}}
            // CHECK-NEXT: to label %invoke.cont{{[0-9]+}} unwind label %[[EH_CLEANUP_OUTER_FINALLY]]
			puts("inner finally");
		}
		return 42;
	}

	@catch(id b) {
        // CHECK: %[[CATCHPAD_B:.*]] = catchpad within %[[CATCHSWITCH_OUTER]]
        // CHECK: to label %invoke.cont{{[0-9]+}} unwind label %[[EH_CLEANUP_OUTER_FINALLY]]
		puts("catch 2");
		return 43;
	}

    // Check that the cleanuppad from the SEH finally funclet was correctly emitted.
    // CHECK: [[EH_CLEANUP_OUTER_FINALLY]]:
    // CHECK: %[[CLEANUP_PAD:.*]] = cleanuppad
    // CHECK: call void @"?fin$0@0@main@@"
    // CHECK: cleanupret from %[[CLEANUP_PAD]] unwind to caller
	@finally {
		puts("cleanup");
	}
	return 0;
}
