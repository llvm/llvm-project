!RUN: %flang -E %s 2>&1 | FileCheck %s
!CHECK: if(.not.(.true.)) error stop "assert(" // ".TRUE." // ") failed " // "at ""
!CHECK-SAME: assert.F90"": " // "7"
#define STR(x) #x
#define POSITION(f,ln) "at "f": " // STR(ln)
#define assert(x) if(.not.(x)) error stop "assert(" // #x // ") failed " // POSITION(__FILE__,__LINE__)
assert(.TRUE.)
end
