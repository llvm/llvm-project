// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fobjc-exceptions -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck %s

void may_throw(void) {
        @throw (id) 1;
}

int main(void) {
        int retval = 0;
        @try {
                may_throw();
                // CHECK: invoke void @may_throw()
                // CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
        }
        // Check that the dispatch block has been emitted correctly.
        // CHECK: [[CATCH_DISPATCH]]:
        // CHECK-NEXT: %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCH_START:.*]] unwind to caller


        // The native WASM EH uses the new exception handling IR instructions
        // (catchswitch, catchpad, etc.) that are also used when targeting Windows MSVC.
        // For SEH, we emit a catchpad instruction for each catch statement. On WASM, we
        // merge all catch statements into one big catch block.

        // CHECK: catchpad within %[[CATCHSWITCH]] [ptr @__objc_id_type_info, ptr null]

        // We use the cxa functions instead of objc_{begin,end}_catch.
        // CHECK: call ptr @__cxa_begin_catch
        @catch(id a) {
            retval = 1;
        }
        @catch(...) {
            retval = 2;
        }
        return retval;
}


