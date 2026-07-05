// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -- -fexceptions

void rethrower() {
    throw;
}

void callsRethrower() {
    rethrower();
}

void callsRethrowerNoexcept() noexcept {
    rethrower();
}

int throwsAndCallsRethrower() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'throwsAndCallsRethrower' which should not throw exceptions
    try {
        throw 1;
    } catch(...) {
        rethrower();
    }
    return 1;
}
// CHECK-MESSAGES: :[[@LINE-6]]:9: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throwsAndCallsRethrower' here

int throwsAndCallsCallsRethrower() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'throwsAndCallsCallsRethrower' which should not throw exceptions
    try {
        throw 1;
    } catch(...) {
        callsRethrower();
    }
    return 1;
}
// CHECK-MESSAGES: :[[@LINE-6]]:9: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throwsAndCallsCallsRethrower' here

void rethrowerNoexcept() noexcept {
    throw;
}
