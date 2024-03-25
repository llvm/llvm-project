// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 %s bugprone-exception-escape %t -- \
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
}

int throwsAndCallsCallsRethrower() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'throwsAndCallsCallsRethrower' which should not throw exceptions
    try {
        throw 1;
    } catch(...) {
        callsRethrower();
    }
}

void rethrowerNoexcept() noexcept {
    throw;
}
